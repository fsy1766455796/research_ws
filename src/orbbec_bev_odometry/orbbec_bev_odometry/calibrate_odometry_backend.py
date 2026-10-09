"""Offline calibration for learned BEV odometry predictions."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


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


def _read_predictions(filename):
    rows = []
    with Path(filename).open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            parsed = {}
            for key, value in row.items():
                parsed[key] = value if key == "sequence" else float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Prediction file is empty: {filename}")
    return rows


def _yaw_features(row):
    forward = float(row["pred_forward_m"])
    left = float(row["pred_left_m"])
    yaw = float(row["pred_yaw_rad"])
    step = math.hypot(forward, left)
    confidence = float(row.get("confidence", 1.0))
    return np.array(
        [
            1.0,
            yaw,
            abs(yaw),
            forward,
            abs(forward),
            left,
            abs(left),
            step,
            confidence,
            yaw * step,
            yaw * confidence,
        ],
        dtype=np.float64,
    )


def _fit_ridge(features, target, ridge):
    gram = features.T @ features
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    return np.linalg.solve(penalty + gram, features.T @ target)


def fit_yaw_calibrator(rows, ridge=1e-3):
    """Fit a validation-only yaw-error model."""
    features = np.stack([_yaw_features(row) for row in rows], axis=0)
    target = np.asarray(
        [
            _wrap(row["pred_yaw_rad"] - row["target_yaw_rad"])
            for row in rows
        ],
        dtype=np.float64,
    )
    coefficients = _fit_ridge(features, target, ridge)
    residual = target - features @ coefficients
    return {
        "coefficients": coefficients,
        "mean_bias_rad": float(np.mean(target)),
        "mean_bias_deg": math.degrees(float(np.mean(target))),
        "residual_rmse_deg": math.degrees(
            float(np.sqrt(np.mean(residual * residual)))
        ),
    }


def apply_correction(rows, calibrator, yaw_gain=1.0, distance_scale=1.0):
    corrected = []
    coefficients = calibrator["coefficients"]
    for row in rows:
        update = dict(row)
        estimated_error = float(_yaw_features(row) @ coefficients)
        update["pred_forward_m"] = float(row["pred_forward_m"]) * distance_scale
        update["pred_left_m"] = float(row["pred_left_m"]) * distance_scale
        update["pred_yaw_rad"] = _wrap(
            float(row["pred_yaw_rad"]) - yaw_gain * estimated_error
        )
        update["estimated_yaw_error_rad"] = estimated_error
        corrected.append(update)
    return corrected


def pair_metrics(rows):
    translation = []
    yaw = []
    predicted_distance = 0.0
    target_distance = 0.0
    false_steps = 0
    for row in rows:
        trans_error = math.hypot(
            row["pred_forward_m"] - row["target_forward_m"],
            row["pred_left_m"] - row["target_left_m"],
        )
        yaw_error = abs(_wrap(row["pred_yaw_rad"] - row["target_yaw_rad"]))
        predicted_step = math.hypot(row["pred_forward_m"], row["pred_left_m"])
        target_step = math.hypot(row["target_forward_m"], row["target_left_m"])
        translation.append(trans_error)
        yaw.append(yaw_error)
        predicted_distance += predicted_step
        target_distance += target_step
        false_steps += int(predicted_step > 0.10 and target_step <= 0.10)
    translation = np.asarray(translation, dtype=np.float64)
    yaw = np.asarray(yaw, dtype=np.float64)
    return {
        "pair_count": int(len(rows)),
        "translation_mae_m": float(np.mean(translation)),
        "translation_rmse_m": float(np.sqrt(np.mean(translation ** 2))),
        "yaw_mae_deg": math.degrees(float(np.mean(yaw))),
        "yaw_rmse_deg": math.degrees(float(np.sqrt(np.mean(yaw ** 2)))),
        "predicted_pair_distance_sum_m": predicted_distance,
        "target_pair_distance_sum_m": target_distance,
        "pair_distance_ratio": (
            predicted_distance / target_distance
            if target_distance > 1e-9 else None
        ),
        "false_step_over_10cm_count": false_steps,
    }


def trajectory_metrics(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    output = {}
    for sequence, sequence_rows in grouped.items():
        by_start = {
            round(row["timestamp_0"], 7): row for row in sequence_rows
        }
        current = min(sequence_rows, key=lambda row: row["timestamp_0"])
        predicted_pose = np.zeros(3, dtype=np.float64)
        target_pose = np.zeros(3, dtype=np.float64)
        errors = [0.0]
        predicted_distance = 0.0
        target_distance = 0.0
        used = 0
        while current is not None:
            predicted_increment = (
                current["pred_forward_m"],
                current["pred_left_m"],
                current["pred_yaw_rad"],
            )
            target_increment = (
                current["target_forward_m"],
                current["target_left_m"],
                current["target_yaw_rad"],
            )
            predicted_distance += math.hypot(
                predicted_increment[0], predicted_increment[1]
            )
            target_distance += math.hypot(
                target_increment[0], target_increment[1]
            )
            predicted_pose = _compose(predicted_pose, predicted_increment)
            target_pose = _compose(target_pose, target_increment)
            errors.append(
                float(np.linalg.norm(predicted_pose[:2] - target_pose[:2]))
            )
            used += 1
            current = by_start.get(round(current["timestamp_1"], 7))
        output[sequence] = {
            "chain_pairs": used,
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
        }
    return output


def _distance_scale(rows):
    predicted = sum(
        math.hypot(row["pred_forward_m"], row["pred_left_m"]) for row in rows
    )
    target = sum(
        math.hypot(row["target_forward_m"], row["target_left_m"]) for row in rows
    )
    return target / predicted if predicted > 1e-9 else 1.0


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--yaw-gain", type=float, default=1.0)
    parser.add_argument("--calibrate-distance-scale", action="store_true")
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    calibration_rows = _read_predictions(args.calibration_predictions)
    test_rows = _read_predictions(args.test_predictions)
    calibrator = fit_yaw_calibrator(calibration_rows, ridge=args.ridge)
    distance_scale = (
        _distance_scale(calibration_rows)
        if args.calibrate_distance_scale else 1.0
    )
    corrected_rows = apply_correction(
        test_rows,
        calibrator,
        yaw_gain=args.yaw_gain,
        distance_scale=distance_scale,
    )
    result = {
        "calibration_predictions": str(
            Path(args.calibration_predictions).resolve()
        ),
        "test_predictions": str(Path(args.test_predictions).resolve()),
        "ridge": args.ridge,
        "yaw_gain": args.yaw_gain,
        "distance_scale": distance_scale,
        "calibrator": {
            "coefficients": calibrator["coefficients"].tolist(),
            "mean_bias_rad": calibrator["mean_bias_rad"],
            "mean_bias_deg": calibrator["mean_bias_deg"],
            "residual_rmse_deg": calibrator["residual_rmse_deg"],
        },
        "raw": {
            "pair": pair_metrics(test_rows),
            "trajectory": trajectory_metrics(test_rows),
        },
        "corrected": {
            "pair": pair_metrics(corrected_rows),
            "trajectory": trajectory_metrics(corrected_rows),
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
