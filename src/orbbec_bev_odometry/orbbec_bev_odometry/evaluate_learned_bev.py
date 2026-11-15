"""Evaluate pairwise error and chained drift of a learned BEV checkpoint."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .learned_bev_dataset import BEVPairDataset
from .learned_bev_model import LearnedBEVOdometry


def _load_checkpoint(filename, device):
    try:
        return torch.load(filename, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=device)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _compose(pose, increment):
    x, y, yaw = pose
    forward, left, dyaw = increment
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            x + cosine * forward - sine * left,
            y + sine * forward + cosine * left,
            _wrap(yaw + dyaw),
        ],
        dtype=np.float64,
    )


def _continuous_segments(rows):
    """Select non-overlapping pairs, retaining the existing 7-digit time grid.

    Prefer the earliest start, then the earliest end; exact duplicate intervals
    keep their input order. Reject invalid intervals before overlap filtering.
    """
    timed_rows = []
    for row in rows:
        start = round(float(row["timestamp_0"]), 7)
        end = round(float(row["timestamp_1"]), 7)
        if not (math.isfinite(start) and math.isfinite(end) and end > start):
            raise ValueError("Pair timestamps must be finite and end after start")
        timed_rows.append((start, end, row))
    segments = []
    endpoint = None
    for start, end, row in sorted(timed_rows, key=lambda item: item[:2]):
        if endpoint is not None and start < endpoint:
            continue
        if endpoint is None or start != endpoint:
            segments.append([])
        segments[-1].append(row)
        endpoint = end
    return segments


def _trajectory_metrics(rows):
    """Pool start-aligned segment errors, including one zero origin per segment.

    Yaw RMSE uses wrapped heading differences at every pose. Cumulative endpoint
    yaw error is signed sum(predicted increments - target increments), unwrapped.
    Coverage counts selected pairs and their duration against all input pairs
    and the full sequence time span, respectively. Endpoints are only meaningful
    per segment, so sequence endpoint scalars are null for multiple segments.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    output = {}
    for sequence, sequence_rows in grouped.items():
        segments = []
        all_errors = []
        all_yaw_errors = []
        for chain in _continuous_segments(sequence_rows):
            metrics, errors, yaw_errors = _segment_metrics(chain)
            segments.append(metrics)
            all_errors.extend(errors)
            all_yaw_errors.extend(yaw_errors)
        used = sum(segment["chain_pairs"] for segment in segments)
        predicted_distance = sum(s["predicted_distance_m"] for s in segments)
        target_distance = sum(s["target_distance_m"] for s in segments)
        covered_duration = sum(s["duration_s"] for s in segments)
        sequence_span = (
            max(round(float(r["timestamp_1"]), 7) for r in sequence_rows)
            - min(round(float(r["timestamp_0"]), 7) for r in sequence_rows)
        )
        output[sequence] = {
            "input_pairs": len(sequence_rows),
            "chain_pairs": used,
            "skipped_pairs": len(sequence_rows) - used,
            "pair_coverage_ratio": used / len(sequence_rows),
            "covered_duration_s": covered_duration,
            "sequence_span_s": sequence_span,
            "time_coverage_ratio": covered_duration / sequence_span,
            "segment_count": len(segments),
            "segments": segments,
            "ate_rmse_m": float(np.sqrt(np.mean(np.square(all_errors)))),
            "yaw_rmse_deg": float(np.sqrt(np.mean(np.square(all_yaw_errors)))),
            "predicted_distance_m": predicted_distance,
            "target_distance_m": target_distance,
            "distance_ratio": (
                predicted_distance / target_distance
                if target_distance > 1e-9 else None
            ),
            "false_step_over_10cm_count": sum(
                s["false_step_over_10cm_count"] for s in segments
            ),
        }
        for field in (
            "endpoint_translation_error_m", "endpoint_yaw_error_deg",
            "endpoint_cumulative_yaw_error_deg",
        ):
            output[sequence][field] = (
                segments[0][field] if len(segments) == 1 else None
            )
    return output


def _segment_metrics(chain):
    """Integrate one continuous segment from a shared zero pose."""
    predicted_pose = np.zeros(3, dtype=np.float64)
    target_pose = np.zeros(3, dtype=np.float64)
    errors = [0.0]
    yaw_errors = [0.0]
    cumulative_yaw_error = 0.0
    predicted_distance = 0.0
    target_distance = 0.0
    false_steps = 0
    for current in chain:
        predicted_step = math.hypot(
            current["pred_forward_m"], current["pred_left_m"]
        )
        target_step = math.hypot(
            current["target_forward_m"], current["target_left_m"]
        )
        predicted_distance += predicted_step
        target_distance += target_step
        false_steps += int(predicted_step > 0.10 and target_step <= 0.10)
        predicted_pose = _compose(
            predicted_pose,
            (current["pred_forward_m"], current["pred_left_m"],
             current["pred_yaw_rad"]),
        )
        target_pose = _compose(
            target_pose,
            (current["target_forward_m"], current["target_left_m"],
             current["target_yaw_rad"]),
        )
        errors.append(float(np.linalg.norm(
            predicted_pose[:2] - target_pose[:2]
        )))
        yaw_errors.append(math.degrees(_wrap(
            predicted_pose[2] - target_pose[2]
        )))
        cumulative_yaw_error += (
            current["pred_yaw_rad"] - current["target_yaw_rad"]
        )
    start = round(float(chain[0]["timestamp_0"]), 7)
    end = round(float(chain[-1]["timestamp_1"]), 7)
    metrics = {
        "timestamp_0": start,
        "timestamp_1": end,
        "duration_s": end - start,
        "chain_pairs": len(chain),
        "ate_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "yaw_rmse_deg": float(np.sqrt(np.mean(np.square(yaw_errors)))),
        "endpoint_translation_error_m": errors[-1],
        "endpoint_yaw_error_deg": abs(math.degrees(_wrap(
            predicted_pose[2] - target_pose[2]
        ))),
        "endpoint_cumulative_yaw_error_deg": math.degrees(
            cumulative_yaw_error
        ),
        "predicted_distance_m": predicted_distance,
        "target_distance_m": target_distance,
        "distance_ratio": (
            predicted_distance / target_distance
            if target_distance > 1e-9 else None
        ),
        "false_step_over_10cm_count": false_steps,
    }
    return metrics, errors, yaw_errors


def build_argument_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None):
    """Run checkpoint evaluation and write CSV/JSON results."""
    args = build_argument_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    model = LearnedBEVOdometry(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = BEVPairDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    with torch.no_grad():
        for batch in loader:
            previous = batch["previous"].to(device, non_blocking=True)
            current = batch["current"].to(device, non_blocking=True)
            output = model(previous, current)
            prediction = output["pose"].cpu().numpy()
            confidence = output["confidence"].cpu().numpy()
            target = batch["target"].numpy()
            for index in range(len(prediction)):
                translation_error = float(np.linalg.norm(
                    prediction[index, :2] - target[index, :2]
                ))
                yaw_error = _wrap(
                    float(prediction[index, 2] - target[index, 2])
                )
                predicted_step = float(np.linalg.norm(prediction[index, :2]))
                target_step = float(np.linalg.norm(target[index, :2]))
                rows.append({
                    "sequence": batch["sequence"][index],
                    "timestamp_0": float(batch["timestamp_0"][index]),
                    "timestamp_1": float(batch["timestamp_1"][index]),
                    "target_forward_m": float(target[index, 0]),
                    "target_left_m": float(target[index, 1]),
                    "target_yaw_rad": float(target[index, 2]),
                    "pred_forward_m": float(prediction[index, 0]),
                    "pred_left_m": float(prediction[index, 1]),
                    "pred_yaw_rad": float(prediction[index, 2]),
                    "translation_error_m": translation_error,
                    "yaw_error_deg": abs(math.degrees(yaw_error)),
                    "predicted_step_m": predicted_step,
                    "target_step_m": target_step,
                    "false_step_over_10cm": int(
                        predicted_step > 0.10 and target_step <= 0.10
                    ),
                    "confidence": float(confidence[index]),
                })

    translation_errors = np.asarray(
        [row["translation_error_m"] for row in rows], dtype=np.float64
    )
    yaw_errors = np.asarray(
        [row["yaw_error_deg"] for row in rows], dtype=np.float64
    )
    predicted_distance = sum(row["predicted_step_m"] for row in rows)
    target_distance = sum(row["target_step_m"] for row in rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "pair_count": len(rows),
        "translation_mae_m": float(np.mean(translation_errors)),
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors ** 2))),
        "translation_p95_m": float(np.percentile(translation_errors, 95)),
        "yaw_mae_deg": float(np.mean(yaw_errors)),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_errors ** 2))),
        "yaw_p95_deg": float(np.percentile(yaw_errors, 95)),
        "predicted_pair_distance_sum_m": predicted_distance,
        "target_pair_distance_sum_m": target_distance,
        "pair_distance_ratio": (
            predicted_distance / target_distance
            if target_distance > 1e-9 else None
        ),
        "false_step_over_10cm_count": sum(
            row["false_step_over_10cm"] for row in rows
        ),
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
