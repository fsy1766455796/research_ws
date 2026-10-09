"""Evaluate learned BEV odometry with a rolling local BEV map."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .learned_bev_dataset import BEVPairDataset
from .learned_bev_model import LearnedBEVOdometry, warp_current_to_previous


def _load_checkpoint(filename, device):
    try:
        return torch.load(filename, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=device)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _matrix(pose):
    forward, left, yaw = pose
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cosine, -sine, forward],
            [sine, cosine, left],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _components(transform):
    return np.array(
        [
            float(transform[0, 2]),
            float(transform[1, 2]),
            _wrap(math.atan2(float(transform[1, 0]), float(transform[0, 0]))),
        ],
        dtype=np.float64,
    )


def _compose(pose, increment):
    return _components(_matrix(pose) @ _matrix(increment))


def _relative_pose(reference_world, frame_world):
    """Return frame coordinates expressed in the reference frame."""
    return _components(np.linalg.inv(_matrix(reference_world)) @ _matrix(frame_world))


def _load_bev(filename):
    with np.load(filename, allow_pickle=False) as archive:
        return np.asarray(archive["bev"], dtype=np.float32)


def _sequence_chains(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record.sequence, []).append(record)
    for sequence, rows in grouped.items():
        by_start = {round(row.timestamp_0, 7): row for row in rows}
        current = min(rows, key=lambda row: row.timestamp_0)
        chain = []
        while current is not None:
            chain.append(current)
            current = by_start.get(round(current.timestamp_1, 7))
        yield sequence, chain


def _build_local_map(frames, reference_pose, metric_bounds, device):
    """Warp recent BEVs into the reference frame and fuse them."""
    if not frames:
        raise ValueError("frames must not be empty")
    warped = []
    for bev, pose in frames:
        relative = _relative_pose(reference_pose, pose)
        tensor = torch.from_numpy(bev).unsqueeze(0).to(device)
        pose_tensor = torch.from_numpy(relative.astype(np.float32)).unsqueeze(0).to(device)
        warped.append(
            warp_current_to_previous(tensor, pose_tensor, metric_bounds)[0]
        )
    stack = torch.stack(warped, dim=0)
    return torch.amax(stack, dim=0)


def _run_model(model, previous_bev, current_bev, device):
    previous = previous_bev.unsqueeze(0).to(device)
    current = torch.from_numpy(current_bev).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(previous, current)
    pose = output["pose"][0].detach().cpu().numpy().astype(np.float64)
    confidence = float(output["confidence"][0].detach().cpu())
    return pose, confidence


def _trajectory_metrics(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    output = {}
    for sequence, sequence_rows in grouped.items():
        predicted_pose = np.zeros(3, dtype=np.float64)
        target_pose = np.zeros(3, dtype=np.float64)
        errors = [0.0]
        predicted_distance = 0.0
        target_distance = 0.0
        false_steps = 0
        for row in sequence_rows:
            predicted_increment = (
                row["pred_forward_m"],
                row["pred_left_m"],
                row["pred_yaw_rad"],
            )
            target_increment = (
                row["target_forward_m"],
                row["target_left_m"],
                row["target_yaw_rad"],
            )
            predicted_step = math.hypot(predicted_increment[0], predicted_increment[1])
            target_step = math.hypot(target_increment[0], target_increment[1])
            predicted_distance += predicted_step
            target_distance += target_step
            false_steps += int(predicted_step > 0.10 and target_step <= 0.10)
            predicted_pose = _compose(predicted_pose, predicted_increment)
            target_pose = _compose(target_pose, target_increment)
            errors.append(float(np.linalg.norm(predicted_pose[:2] - target_pose[:2])))
        output[sequence] = {
            "chain_pairs": len(sequence_rows),
            "ate_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
            "endpoint_translation_error_m": errors[-1],
            "endpoint_yaw_error_deg": abs(
                math.degrees(_wrap(predicted_pose[2] - target_pose[2]))
            ),
            "predicted_distance_m": predicted_distance,
            "target_distance_m": target_distance,
            "distance_ratio": (
                predicted_distance / target_distance
                if target_distance > 1e-9 else None
            ),
            "false_step_over_10cm_count": false_steps,
        }
    return output


def _pair_metrics(rows):
    translation = []
    yaw = []
    predicted_distance = 0.0
    target_distance = 0.0
    false_steps = 0
    for row in rows:
        translation.append(
            math.hypot(
                row["pred_forward_m"] - row["target_forward_m"],
                row["pred_left_m"] - row["target_left_m"],
            )
        )
        yaw.append(abs(_wrap(row["pred_yaw_rad"] - row["target_yaw_rad"])))
        predicted_step = math.hypot(row["pred_forward_m"], row["pred_left_m"])
        target_step = math.hypot(row["target_forward_m"], row["target_left_m"])
        predicted_distance += predicted_step
        target_distance += target_step
        false_steps += int(predicted_step > 0.10 and target_step <= 0.10)
    translation = np.asarray(translation, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    return {
        "pair_count": len(rows),
        "translation_mae_m": float(np.mean(translation)),
        "translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
        "translation_p95_m": float(np.percentile(translation, 95)),
        "yaw_mae_deg": math.degrees(float(np.mean(yaw))),
        "yaw_rmse_deg": math.degrees(float(np.sqrt(np.mean(yaw ** 2)))),
        "yaw_p95_deg": math.degrees(float(np.percentile(yaw, 95))),
        "predicted_pair_distance_sum_m": predicted_distance,
        "target_pair_distance_sum_m": target_distance,
        "pair_distance_ratio": (
            predicted_distance / target_distance
            if target_distance > 1e-9 else None
        ),
        "false_step_over_10cm_count": false_steps,
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--warmup-pairwise", type=int, default=1)
    parser.add_argument(
        "--local-weight",
        type=float,
        default=1.0,
        help="Blend factor for local-map prediction after warmup.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model = LearnedBEVOdometry(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metric_bounds = tuple(checkpoint["model_config"]["metric_bounds"])
    dataset = BEVPairDataset(args.manifest)

    rows = []
    for sequence, chain in _sequence_chains(dataset.records):
        recent_frames = []
        predicted_pose = np.zeros(3, dtype=np.float64)
        previous_bev = _load_bev(chain[0].bev_0)
        recent_frames.append((previous_bev, predicted_pose.copy()))
        for index, record in enumerate(chain):
            current_bev = _load_bev(record.bev_1)
            pairwise_reference = torch.from_numpy(recent_frames[-1][0]).to(device)
            pairwise_prediction, pairwise_confidence = _run_model(
                model, pairwise_reference, current_bev, device
            )
            if index < args.warmup_pairwise or len(recent_frames) == 1:
                prediction = pairwise_prediction
                confidence = pairwise_confidence
            else:
                reference_bev = _build_local_map(
                    recent_frames[-args.window_size:],
                    predicted_pose,
                    metric_bounds,
                    device,
                )
                local_prediction, local_confidence = _run_model(
                    model, reference_bev, current_bev, device
                )
                local_weight = min(max(float(args.local_weight), 0.0), 1.0)
                prediction = (
                    (1.0 - local_weight) * pairwise_prediction
                    + local_weight * local_prediction
                )
                prediction[2] = _wrap(
                    (1.0 - local_weight) * pairwise_prediction[2]
                    + local_weight * local_prediction[2]
                )
                confidence = (
                    (1.0 - local_weight) * pairwise_confidence
                    + local_weight * local_confidence
                )
            predicted_pose = _compose(predicted_pose, prediction)
            target = record.target.astype(np.float64)
            rows.append(
                {
                    "sequence": sequence,
                    "timestamp_0": record.timestamp_0,
                    "timestamp_1": record.timestamp_1,
                    "target_forward_m": float(target[0]),
                    "target_left_m": float(target[1]),
                    "target_yaw_rad": float(target[2]),
                    "pred_forward_m": float(prediction[0]),
                    "pred_left_m": float(prediction[1]),
                    "pred_yaw_rad": float(prediction[2]),
                    "confidence": confidence,
                }
            )
            recent_frames.append((current_bev, predicted_pose.copy()))
            if len(recent_frames) > args.window_size:
                recent_frames.pop(0)

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "window_size": args.window_size,
        "warmup_pairwise": args.warmup_pairwise,
        "local_weight": args.local_weight,
        "pair": _pair_metrics(rows),
        "trajectory": _trajectory_metrics(rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
