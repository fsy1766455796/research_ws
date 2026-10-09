"""Evaluate a planar ICP baseline on cached BEV pairs."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .evaluate_learned_bev import _trajectory_metrics, _wrap
from .openloris_data import load_pair_manifest
from .planar_icp import IcpConfig, PlanarPointToPlaneICP
from .se2 import components


CHANNELS = {
    "occupancy": 0,
    "density": 1,
    "height_mean": 2,
    "height_max": 3,
    "height_variance": 4,
}


def _load_points(filename, channel, threshold, max_points, seed):
    with np.load(filename, allow_pickle=False) as archive:
        bev = np.asarray(archive["bev"], dtype=np.float32)
    image = np.clip(bev[int(channel)], 0.0, 1.0)
    rows, cols = np.nonzero(image > float(threshold))
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float64)
    forward = -1.0 + (rows.astype(np.float64) + 0.5) * 0.05
    left = -3.5 + (cols.astype(np.float64) + 0.5) * 0.05
    points = np.column_stack((forward, left))
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(points), size=max_points, replace=False)
        points = points[np.sort(index)]
    return np.ascontiguousarray(points, dtype=np.float64)


def _estimate_pair(record, solver, channel, threshold, max_points, index, mode):
    previous = _load_points(record.bev_0, channel, threshold, max_points, index * 2)
    current = _load_points(record.bev_1, channel, threshold, max_points, index * 2 + 1)
    initial = np.eye(3, dtype=np.float64)
    if mode == "multiscale":
        result = solver.register_multiscale(current, previous, initial=initial)
    else:
        result = solver.register_point_to_point(current, previous, initial=initial)
    if result.accepted:
        forward, left, yaw = components(result.transform)
    else:
        forward, left, yaw = 0.0, 0.0, 0.0
    return {
        "pred_forward_m": float(forward),
        "pred_left_m": float(left),
        "pred_yaw_rad": float(yaw),
        "accepted": int(result.accepted),
        "rmse": float(result.rmse),
        "correspondences": int(result.correspondences),
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--channel", choices=tuple(CHANNELS), default="density")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--mode", choices=("point_to_point", "multiscale"), default="point_to_point")
    parser.add_argument("--max-correspondence", type=float, default=0.35)
    parser.add_argument("--min-correspondences", type=int, default=35)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    records = load_pair_manifest(args.manifest)
    if not records:
        raise ValueError(f"Pair manifest is empty: {args.manifest}")
    solver = PlanarPointToPlaneICP(IcpConfig(
        max_correspondence=args.max_correspondence,
        min_correspondences=args.min_correspondences,
    ))

    rows = []
    for index, record in enumerate(records):
        prediction = _estimate_pair(
            record,
            solver,
            CHANNELS[args.channel],
            args.threshold,
            args.max_points,
            index,
            args.mode,
        )
        target = record.target
        translation_error = math.hypot(
            prediction["pred_forward_m"] - float(target[0]),
            prediction["pred_left_m"] - float(target[1]),
        )
        yaw_error = _wrap(prediction["pred_yaw_rad"] - float(target[2]))
        predicted_step = math.hypot(
            prediction["pred_forward_m"], prediction["pred_left_m"]
        )
        target_step = math.hypot(float(target[0]), float(target[1]))
        rows.append({
            "sequence": record.sequence,
            "timestamp_0": record.timestamp_0,
            "timestamp_1": record.timestamp_1,
            "target_forward_m": float(target[0]),
            "target_left_m": float(target[1]),
            "target_yaw_rad": float(target[2]),
            **prediction,
            "translation_error_m": translation_error,
            "yaw_error_deg": abs(math.degrees(yaw_error)),
            "predicted_step_m": predicted_step,
            "target_step_m": target_step,
            "false_step_over_10cm": int(
                predicted_step > 0.10 and target_step <= 0.10
            ),
        })
        if (index + 1) % 250 == 0:
            print(f"processed {index + 1}/{len(records)}", flush=True)

    translation_errors = np.asarray(
        [row["translation_error_m"] for row in rows], dtype=np.float64
    )
    yaw_errors = np.asarray(
        [row["yaw_error_deg"] for row in rows], dtype=np.float64
    )
    predicted_distance = sum(row["predicted_step_m"] for row in rows)
    target_distance = sum(row["target_step_m"] for row in rows)
    summary = {
        "method": "planar_icp",
        "manifest": str(Path(args.manifest).resolve()),
        "channel": args.channel,
        "mode": args.mode,
        "pair_count": len(rows),
        "accepted_count": sum(row["accepted"] for row in rows),
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
