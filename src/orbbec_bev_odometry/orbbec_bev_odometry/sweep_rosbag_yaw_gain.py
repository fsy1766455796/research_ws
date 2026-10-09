"""Offline yaw-gain sweep for learned-BEV rosbag predictions."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _read_measurements(filename):
    measurements = []
    with Path(filename).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            measurements.append(
                (
                    float(row["pred_forward_m"]),
                    float(row["pred_left_m"]),
                    float(row["pred_yaw_rad"]),
                )
            )
    return np.asarray(measurements, dtype=np.float64)


def _compose_all(measurements, yaw_gain):
    pose = np.zeros(3, dtype=np.float64)
    poses = [pose.copy()]
    for forward, left, yaw_step in measurements:
        yaw_step *= float(yaw_gain)
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        pose = np.asarray(
            [
                pose[0] + cosine * forward - sine * left,
                pose[1] + sine * forward + cosine * left,
                _wrap(pose[2] + yaw_step),
            ],
            dtype=np.float64,
        )
        poses.append(pose.copy())
    return np.asarray(poses, dtype=np.float64)


def _summary(poses):
    endpoint = poses[-1]
    return {
        "path_chord_m": float(np.linalg.norm(endpoint[:2])),
        "endpoint_x_m": float(endpoint[0]),
        "endpoint_y_m": float(endpoint[1]),
        "endpoint_yaw_deg": math.degrees(float(endpoint[2])),
    }


def _parse_values(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def _camera_yaw_from_summary(directory):
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open(encoding="utf-8") as stream:
        return json.load(stream).get("camera_yaw_deg")


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--pattern", default="sweep_z_0.5_yaw_fine_*")
    parser.add_argument(
        "--yaw-gains",
        default="1.0,1.2,1.4,1.5,1.55,1.6,1.7,2.0",
        help="Comma-separated yaw gain values.",
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--yaw-score-weight",
        type=float,
        default=0.01,
        help="Score = endpoint distance + this * abs(endpoint yaw deg).",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    root = Path(args.results_root)
    yaw_gains = _parse_values(args.yaw_gains)
    rows = []
    for directory in sorted(root.glob(args.pattern)):
        predictions = directory / "predictions.csv"
        if not predictions.is_file():
            continue
        camera_yaw = _camera_yaw_from_summary(directory)
        measurements = _read_measurements(predictions)
        for yaw_gain in yaw_gains:
            trajectory = _compose_all(measurements, yaw_gain)
            summary = _summary(trajectory)
            rows.append(
                {
                    "source_dir": str(directory),
                    "camera_yaw_deg": camera_yaw,
                    "yaw_gain": yaw_gain,
                    **summary,
                    "abs_endpoint_yaw_deg": abs(summary["endpoint_yaw_deg"]),
                    "score": summary["path_chord_m"]
                    + args.yaw_score_weight * abs(summary["endpoint_yaw_deg"]),
                }
            )
    rows.sort(key=lambda row: row["score"])
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "score",
            "camera_yaw_deg",
            "yaw_gain",
            "path_chord_m",
            "endpoint_yaw_deg",
            "abs_endpoint_yaw_deg",
            "endpoint_x_m",
            "endpoint_y_m",
            "source_dir",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_csv": str(output), "best": rows[:10]}, indent=2))


if __name__ == "__main__":
    main()
