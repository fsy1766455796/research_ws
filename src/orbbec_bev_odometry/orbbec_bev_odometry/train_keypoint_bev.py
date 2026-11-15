"""Train / evaluate the keypoint + weighted-Procrustes BEV odometry (route A).

Checkpoint selection uses the validation manifest only: chained-trajectory
ATE (same rules as evaluate_learned_bev) plus a small yaw term.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .evaluate_learned_bev import _trajectory_metrics, _wrap
from .keypoint_bev_loss import keypoint_bev_loss
from .keypoint_bev_model import KeypointBEVOdometry
from .learned_bev_dataset import BEVPairDataset


def predict_rows(model, loader, device):
    """Run the model and return rows in the evaluate_learned_bev CSV schema."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(batch["previous"].to(device, non_blocking=True),
                               batch["current"].to(device, non_blocking=True))
            prediction = output["pose"].float().cpu().numpy()
            confidence = output["confidence"].float().cpu().numpy()
            target = batch["target"].numpy()
            for i in range(len(prediction)):
                predicted_step = float(np.linalg.norm(prediction[i, :2]))
                target_step = float(np.linalg.norm(target[i, :2]))
                rows.append({
                    "sequence": batch["sequence"][i],
                    "timestamp_0": float(batch["timestamp_0"][i]),
                    "timestamp_1": float(batch["timestamp_1"][i]),
                    "target_forward_m": float(target[i, 0]),
                    "target_left_m": float(target[i, 1]),
                    "target_yaw_rad": float(target[i, 2]),
                    "pred_forward_m": float(prediction[i, 0]),
                    "pred_left_m": float(prediction[i, 1]),
                    "pred_yaw_rad": float(prediction[i, 2]),
                    "translation_error_m": float(np.linalg.norm(prediction[i, :2] - target[i, :2])),
                    "yaw_error_deg": abs(math.degrees(_wrap(float(prediction[i, 2] - target[i, 2])))),
                    "predicted_step_m": predicted_step,
                    "target_step_m": target_step,
                    "false_step_over_10cm": int(predicted_step > 0.10 and target_step <= 0.10),
                    "confidence": float(confidence[i]),
                })
    return rows


def summarize_rows(rows):
    translation = np.array([r["translation_error_m"] for r in rows])
    yaw = np.array([r["yaw_error_deg"] for r in rows])
    predicted = sum(r["predicted_step_m"] for r in rows)
    target = sum(r["target_step_m"] for r in rows)
    trajectory = _trajectory_metrics(rows)
    ate = [m["ate_rmse_m"] for m in trajectory.values()]
    traj_yaw = [m["yaw_rmse_deg"] for m in trajectory.values()]
    return {
        "pair_count": len(rows),
        "translation_mae_m": float(translation.mean()),
        "translation_rmse_m": float(np.sqrt((translation ** 2).mean())),
        "yaw_mae_deg": float(yaw.mean()),
        "yaw_rmse_deg": float(np.sqrt((yaw ** 2).mean())),
        "pair_distance_ratio": predicted / target if target > 1e-9 else None,
        "false_step_over_10cm_count": int(sum(r["false_step_over_10cm"] for r in rows)),
        "trajectory_ate_rmse_m_mean": float(np.mean(ate)),
        "trajectory_yaw_rmse_deg_mean": float(np.mean(traj_yaw)),
        "trajectory": trajectory,
    }


def selection_score(summary):
    return summary["trajectory_ate_rmse_m_mean"] + 0.01 * summary["trajectory_yaw_rmse_deg_mean"]


def write_outputs(rows, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_model(config, device):
    return KeypointBEVOdometry(**config).to(device)


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--train-manifest", required=True)
    train.add_argument("--validation-manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--static-sample-weight", type=float, default=2.0)
    train.add_argument("--warmup-uniform-epochs", type=int, default=2,
                       help="Epochs with learned validity disabled (official GKP idea)")
    train.add_argument("--yaw-weight", type=float, default=4.0)
    train.add_argument("--correspondence-weight", type=float, default=0.2)
    train.add_argument("--first-pass-weight", type=float, default=0.5)
    train.add_argument("--iterations", type=int, default=2)
    train.add_argument("--keypoint-patch", type=int, default=2)
    train.add_argument("--limit-train", type=int, default=0, help="Use only N train pairs (overfit check)")
    train.add_argument("--limit-validation", type=int, default=0)
    train.add_argument("--pretrained", help="Initialise weights from a keypoint checkpoint")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--workers", type=int, default=2)
    return parser


def _loader(dataset, batch_size, workers, device, sampler=None, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=workers > 0)


def train(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a new empty output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = BEVPairDataset(args.train_manifest, reverse_probability=0.5)
    validation_set = BEVPairDataset(args.validation_manifest)
    records = train_set.records
    if args.limit_train:
        indices = list(range(0, len(train_set), max(len(train_set) // args.limit_train, 1)))[:args.limit_train]
        train_set, records = Subset(train_set, indices), [records[i] for i in indices]
    if args.limit_validation:
        validation_set = Subset(validation_set, list(range(min(args.limit_validation, len(validation_set)))))
    sample = (train_set[0])["previous"]
    model_config = dict(input_channels=int(sample.shape[0]), bev_height=int(sample.shape[1]),
                        bev_width=int(sample.shape[2]), iterations=args.iterations,
                        keypoint_patch=args.keypoint_patch)
    model = build_model(model_config, device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=device, weights_only=False)["model_state"])
    loss_config = dict(yaw_weight=args.yaw_weight, correspondence_weight=args.correspondence_weight,
                       first_pass_weight=args.first_pass_weight)
    weights = np.array([args.static_sample_weight if (math.hypot(r.forward_m, r.left_m) < 0.015
                        and abs(r.yaw_rad) < math.radians(0.3)) else 1.0 for r in records])
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    train_loader = _loader(train_set, args.batch_size, args.workers, device, sampler=sampler)
    validation_loader = _loader(validation_set, args.batch_size, args.workers, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate, total_steps=args.epochs * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(device=str(device), train_pairs=len(train_set), validation_pairs=len(validation_set),
                model_config=model_config, loss_config=loss_config, args=vars(args))
    (output_dir / "config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    best = float("inf")
    for epoch in range(args.epochs):
        started = time.time()
        model.uniform_validity = epoch < args.warmup_uniform_epochs
        model.train()
        sums, count, skipped = {}, 0, 0
        for batch in train_loader:
            previous = batch["previous"].to(device, non_blocking=True)
            current = batch["current"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(previous, current)
            parts = keypoint_bev_loss(output, target, **loss_config)
            if not torch.isfinite(parts["loss"]):
                skipped += 1
                scheduler.step()
                continue
            scaler.scale(parts["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            n = len(target)
            count += n
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
        model.uniform_validity = False
        rows = predict_rows(model, validation_loader, device)
        validation = summarize_rows(rows)
        score = selection_score(validation)
        record = dict(epoch=epoch, seconds=time.time() - started, lr=optimizer.param_groups[0]["lr"],
                      skipped_nonfinite_batches=skipped,
                      train={k: v / max(count, 1) for k, v in sums.items()},
                      validation={k: v for k, v in validation.items() if k != "trajectory"},
                      validation_score=score, temperature=float(model.temperature()))
        print(json.dumps(record), flush=True)
        with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = dict(epoch=epoch, model_type="keypoint_bev", model_config=model_config,
                          model_state=model.state_dict(), loss_config=loss_config,
                          validation_score=score, train_manifest=str(Path(args.train_manifest).resolve()),
                          validation_manifest=str(Path(args.validation_manifest).resolve()))
        torch.save(checkpoint, output_dir / "last.pt")
        if score < best:
            best = score
            torch.save(checkpoint, output_dir / "best.pt")
            write_outputs(rows, validation, output_dir / "best_validation")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["model_state"])
    dataset = BEVPairDataset(args.manifest)
    rows = predict_rows(model, _loader(dataset, args.batch_size, args.workers, device), device)
    summary = summarize_rows(rows)
    summary.update(checkpoint=str(Path(args.checkpoint).resolve()), manifest=str(Path(args.manifest).resolve()),
                   checkpoint_epoch=checkpoint["epoch"])
    write_outputs(rows, summary, args.output_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "trajectory"}, indent=2))


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    train(args) if args.command == "train" else evaluate(args)


if __name__ == "__main__":
    main()
