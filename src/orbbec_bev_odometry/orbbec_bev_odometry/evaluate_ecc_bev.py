"""Evaluate a classical ECC BEV registration baseline on cached BEV pairs."""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .evaluate_learned_bev import _trajectory_metrics, _wrap
from .openloris_data import load_pair_manifest


CHANNELS = {
    "occupancy": 0,
    "density": 1,
    "height_mean": 2,
    "height_max": 3,
    "height_variance": 4,
}


def _estimate_translation_mm(previous_bev, current_bev, resolution_mm):
    shift_px, response = cv2.phaseCorrelate(
        previous_bev.astype(np.float32),
        current_bev.astype(np.float32),
    )
    shift_u_px, shift_v_px = shift_px
    dx_mm = shift_u_px * resolution_mm
    dz_mm = -shift_v_px * resolution_mm
    return dx_mm, dz_mm, response


def _estimate_motion_mm(
    previous_bev,
    current_bev,
    x_min_mm,
    z_min_mm,
    resolution_mm,
    ecc_iterations,
    ecc_epsilon,
):
    initial_dx_mm, initial_dz_mm, initial_response = _estimate_translation_mm(
        previous_bev,
        current_bev,
        resolution_mm,
    )
    warp = np.array(
        [
            [1.0, 0.0, initial_dx_mm / resolution_mm],
            [0.0, 1.0, -initial_dz_mm / resolution_mm],
        ],
        dtype=np.float32,
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(ecc_iterations),
        float(ecc_epsilon),
    )
    previous_float = previous_bev.astype(np.float32)
    current_float = current_bev.astype(np.float32)
    current_mask = (current_bev > 1e-4).astype(np.uint8) * 255
    current_mask = cv2.dilate(
        current_mask, np.ones((5, 5), dtype=np.uint8), iterations=1
    )

    try:
        response, warp = cv2.findTransformECC(
            previous_float,
            current_float,
            warp,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            current_mask,
            5,
        )
    except cv2.error:
        return initial_dx_mm, initial_dz_mm, 0.0, initial_response, warp

    height = previous_bev.shape[0]
    origin_u = -float(x_min_mm) / resolution_mm
    origin_v = (height - 1) + float(z_min_mm) / resolution_mm
    origin_u_warped = (
        float(warp[0, 0]) * origin_u
        + float(warp[0, 1]) * origin_v
        + float(warp[0, 2])
    )
    origin_v_warped = (
        float(warp[1, 0]) * origin_u
        + float(warp[1, 1]) * origin_v
        + float(warp[1, 2])
    )
    yaw_step_rad = math.atan2(float(warp[0, 1]), float(warp[0, 0]))
    dx_mm = (origin_u_warped - origin_u) * resolution_mm
    dz_mm = -(origin_v_warped - origin_v) * resolution_mm
    return dx_mm, dz_mm, yaw_step_rad, float(response), warp


def _load_ecc_image(filename, channel):
    with np.load(filename, allow_pickle=False) as archive:
        bev = np.asarray(archive["bev"], dtype=np.float32)
    image = np.clip(bev[int(channel)], 0.0, 1.0)
    # The online ECC path displays forward as image-up. Cached learned BEVs
    # store rows increasing forward, so flip them before using the same core.
    return np.flipud(image).astype(np.float32)


def _estimate_pair(record, channel, resolution_m, yaw_sign, forward_sign, left_sign,
                   ecc_iterations, ecc_epsilon):
    previous = _load_ecc_image(record.bev_0, channel)
    current = _load_ecc_image(record.bev_1, channel)
    dx_mm, dz_mm, yaw_rad, response, _ = _estimate_motion_mm(
        previous,
        current,
        x_min_mm=-3500.0,
        z_min_mm=-1000.0,
        resolution_mm=float(resolution_m) * 1000.0,
        ecc_iterations=ecc_iterations,
        ecc_epsilon=ecc_epsilon,
    )
    return {
        "pred_forward_m": float(forward_sign) * dz_mm * 0.001,
        "pred_left_m": float(left_sign) * dx_mm * 0.001,
        "pred_yaw_rad": float(yaw_sign) * yaw_rad,
        "response": float(response),
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--channel", choices=tuple(CHANNELS), default="density")
    parser.add_argument("--resolution-m", type=float, default=0.05)
    parser.add_argument("--yaw-sign", type=float, default=1.0)
    parser.add_argument("--forward-sign", type=float, default=1.0)
    parser.add_argument("--left-sign", type=float, default=1.0)
    parser.add_argument("--ecc-iterations", type=int, default=50)
    parser.add_argument("--ecc-epsilon", type=float, default=1e-4)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    records = load_pair_manifest(args.manifest)
    if not records:
        raise ValueError(f"Pair manifest is empty: {args.manifest}")

    rows = []
    for record in records:
        prediction = _estimate_pair(
            record,
            CHANNELS[args.channel],
            args.resolution_m,
            args.yaw_sign,
            args.forward_sign,
            args.left_sign,
            args.ecc_iterations,
            args.ecc_epsilon,
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

    translation_errors = np.asarray(
        [row["translation_error_m"] for row in rows], dtype=np.float64
    )
    yaw_errors = np.asarray(
        [row["yaw_error_deg"] for row in rows], dtype=np.float64
    )
    predicted_distance = sum(row["predicted_step_m"] for row in rows)
    target_distance = sum(row["target_step_m"] for row in rows)
    summary = {
        "method": "ecc",
        "manifest": str(Path(args.manifest).resolve()),
        "channel": args.channel,
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
        "yaw_sign": args.yaw_sign,
        "forward_sign": args.forward_sign,
        "left_sign": args.left_sign,
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
