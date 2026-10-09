"""Train the height-aware BEV SE(2) registration model."""

import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .learned_bev_dataset import BEVPairDataset
from .learned_bev_loss import learned_bev_loss, pose_error_metrics
from .learned_bev_model import LearnedBEVOdometry


def _device_from_argument(value):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _accumulate_metrics(total, update):
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + value


def _summarize(total):
    count = max(int(total.get("count", 0)), 1)
    target_distance = total.get("target_distance_sum_m", 0.0)
    return {
        "translation_mae_m": total["translation_sum_m"] / count,
        "translation_rmse_m": math.sqrt(
            total["translation_squared_sum_m2"] / count
        ),
        "yaw_mae_deg": math.degrees(total["yaw_sum_rad"] / count),
        "yaw_rmse_deg": math.degrees(
            math.sqrt(total["yaw_squared_sum_rad2"] / count)
        ),
        "predicted_distance_sum_m": total.get(
            "predicted_distance_sum_m", 0.0
        ),
        "target_distance_sum_m": target_distance,
        "distance_ratio": (
            total.get("predicted_distance_sum_m", 0.0) / target_distance
            if target_distance > 1e-9 else None
        ),
        "false_step_over_10cm_count": int(
            total.get("false_step_over_10cm_count", 0)
        ),
        "count": count,
    }


def _run_epoch(model, loader, device, loss_weights, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    loss_sum = 0.0
    amp_enabled = device.type == "cuda"
    for batch in loader:
        previous = batch["previous"].to(device, non_blocking=True)
        current = batch["current"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(previous, current)
            losses = learned_bev_loss(output, target, **loss_weights)
            if training:
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
        batch_size = int(len(target))
        loss_sum += float(losses["loss"].detach().item()) * batch_size
        _accumulate_metrics(
            totals,
            pose_error_metrics(output["pose"].detach(), target.detach()),
        )
    summary = _summarize(totals)
    summary["loss"] = loss_sum / max(summary["count"], 1)
    return summary


def build_argument_parser():
    """Build the training command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--static-sample-weight", type=float, default=2.0)
    parser.add_argument("--max-translation-m", type=float, default=0.8)
    parser.add_argument("--max-yaw-deg", type=float, default=20.0)
    parser.add_argument("--yaw-bins", type=int, default=41)
    parser.add_argument("--forward-min-m", type=float, default=-1.0)
    parser.add_argument("--forward-max-m", type=float, default=6.0)
    parser.add_argument("--left-min-m", type=float, default=-3.5)
    parser.add_argument("--left-max-m", type=float, default=3.5)
    parser.add_argument("--translation-weight", type=float, default=1.0)
    parser.add_argument("--yaw-weight", type=float, default=2.0)
    parser.add_argument("--correlation-weight", type=float, default=0.25)
    parser.add_argument("--static-weight", type=float, default=0.5)
    parser.add_argument("--step-length-weight", type=float, default=0.5)
    parser.add_argument("--confidence-weight", type=float, default=0.05)
    parser.add_argument(
        "--best-score",
        choices=("translation", "combined"),
        default="translation",
        help=(
            "Metric for best.pt. combined also penalizes yaw RMSE and "
            "distance-ratio error."
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        help="Load only model weights before starting a fresh training run.",
    )
    parser.add_argument("--resume")
    return parser


def _checkpoint_score(summary, mode):
    if mode == "translation":
        return summary["translation_rmse_m"]
    distance_ratio = summary.get("distance_ratio")
    distance_penalty = (
        abs(float(distance_ratio) - 1.0)
        if distance_ratio is not None else 1.0
    )
    return (
        summary["translation_rmse_m"]
        + 0.01 * summary["yaw_rmse_deg"]
        + 0.02 * distance_penalty
    )


def main(argv=None):
    """Train the model and save last/best checkpoints."""
    args = build_argument_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _device_from_argument(args.device)
    train_dataset = BEVPairDataset(
        args.train_manifest, reverse_probability=0.5
    )
    validation_dataset = BEVPairDataset(args.validation_manifest)
    sample = train_dataset[0]["previous"]
    model_config = {
        "input_channels": int(sample.shape[0]),
        "metric_bounds": (
            args.forward_min_m,
            args.forward_max_m,
            args.left_min_m,
            args.left_max_m,
        ),
        "max_translation_m": args.max_translation_m,
        "max_yaw_deg": args.max_yaw_deg,
        "yaw_bins": args.yaw_bins,
        "bev_height": int(sample.shape[1]),
        "bev_width": int(sample.shape[2]),
    }
    model = LearnedBEVOdometry(**model_config).to(device)
    if args.pretrained_checkpoint:
        pretrained = torch.load(args.pretrained_checkpoint, map_location=device)
        pretrained_config = pretrained.get("model_config", {})
        if pretrained_config and pretrained_config != model_config:
            print(
                json.dumps(
                    {
                        "warning": (
                            "pretrained model_config differs from current "
                            "model_config; loading state_dict strictly"
                        ),
                        "pretrained_model_config": pretrained_config,
                        "current_model_config": model_config,
                    }
                )
            )
        model.load_state_dict(pretrained["model_state"])
    loss_weights = {
        "translation_weight": args.translation_weight,
        "yaw_weight": args.yaw_weight,
        "correlation_weight": args.correlation_weight,
        "static_weight": args.static_weight,
        "step_length_weight": args.step_length_weight,
        "confidence_weight": args.confidence_weight,
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=device.type == "cuda"
        )
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    best_validation = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("best_validation", best_validation))

    sample_weights = np.ones(len(train_dataset), dtype=np.float64)
    static_count = 0
    for index, record in enumerate(train_dataset.records):
        is_static = (
            math.hypot(record.forward_m, record.left_m) < 0.015
            and abs(record.yaw_rad) < math.radians(0.3)
        )
        if is_static:
            sample_weights[index] = args.static_sample_weight
            static_count += 1
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    preprocessing_metadata = {}
    metadata_path = Path(args.train_manifest).parent / "metadata.json"
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as stream:
            preprocessing_metadata = json.load(stream)
    print(
        json.dumps(
            {
                "device": str(device),
                "train_pairs": len(train_dataset),
                "train_static_pairs": static_count,
                "validation_pairs": len(validation_dataset),
                "model_config": model_config,
                "loss_weights": loss_weights,
                "best_score": args.best_score,
                "pretrained_checkpoint": args.pretrained_checkpoint,
            },
            indent=2,
        )
    )

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_summary = _run_epoch(
            model,
            train_loader,
            device,
            loss_weights,
            optimizer=optimizer,
            scaler=scaler,
        )
        with torch.no_grad():
            validation_summary = _run_epoch(
                model, validation_loader, device, loss_weights
            )
        scheduler.step()
        epoch_summary = {
            "epoch": epoch,
            "seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_summary,
            "validation": validation_summary,
        }
        print(json.dumps(epoch_summary))
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch_summary) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_validation": best_validation,
            "best_score": args.best_score,
            "loss_weights": loss_weights,
            "pretrained_checkpoint": args.pretrained_checkpoint,
            "train_manifest": str(Path(args.train_manifest).resolve()),
            "validation_manifest": str(
                Path(args.validation_manifest).resolve()
            ),
            "bev_config": preprocessing_metadata.get("bev_config", {}),
            "preprocessing_metadata": preprocessing_metadata,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        score = _checkpoint_score(validation_summary, args.best_score)
        if score < best_validation:
            best_validation = score
            checkpoint["best_validation"] = best_validation
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
