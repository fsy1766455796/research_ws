"""Pose-graph backend for learned BEV odometry prediction CSV files."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def _wrap_np(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _wrap_torch(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


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


def _compose_np(pose, increment):
    x, y, yaw = pose
    forward, left, dyaw = increment
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            x + cosine * forward - sine * left,
            y + sine * forward + cosine * left,
            _wrap_np(yaw + dyaw),
        ],
        dtype=np.float64,
    )


def _relative_np(pose_i, pose_j):
    xi, yi, yawi = pose_i
    xj, yj, yawj = pose_j
    dx = xj - xi
    dy = yj - yi
    cosine, sine = math.cos(yawi), math.sin(yawi)
    return np.array(
        [
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            _wrap_np(yawj - yawi),
        ],
        dtype=np.float64,
    )


def _relative_torch(pose_i, pose_j):
    dx = pose_j[..., 0] - pose_i[..., 0]
    dy = pose_j[..., 1] - pose_i[..., 1]
    yaw = pose_i[..., 2]
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    return torch.stack(
        (
            cosine * dx + sine * dy,
            -sine * dx + cosine * dy,
            _wrap_torch(pose_j[..., 2] - pose_i[..., 2]),
        ),
        dim=-1,
    )


def _integrate(rows):
    poses = [np.zeros(3, dtype=np.float64)]
    target_poses = [np.zeros(3, dtype=np.float64)]
    for row in rows:
        poses.append(
            _compose_np(
                poses[-1],
                (
                    row["pred_forward_m"],
                    row["pred_left_m"],
                    row["pred_yaw_rad"],
                ),
            )
        )
        target_poses.append(
            _compose_np(
                target_poses[-1],
                (
                    row["target_forward_m"],
                    row["target_left_m"],
                    row["target_yaw_rad"],
                ),
            )
        )
    return np.asarray(poses), np.asarray(target_poses)


def _main_chain(rows):
    by_start = {round(row["timestamp_0"], 7): row for row in rows}
    current = min(rows, key=lambda row: row["timestamp_0"])
    chain = []
    while current is not None:
        chain.append(current)
        current = by_start.get(round(current["timestamp_1"], 7))
    return chain


def _trajectory_summary(poses, target_poses):
    errors = np.linalg.norm(poses[:, :2] - target_poses[:, :2], axis=1)
    endpoint_yaw = abs(_wrap_np(float(poses[-1, 2] - target_poses[-1, 2])))
    return {
        "node_count": int(len(poses)),
        "ate_rmse_m": float(np.sqrt(np.mean(errors * errors))),
        "endpoint_translation_error_m": float(errors[-1]),
        "endpoint_yaw_error_deg": math.degrees(endpoint_yaw),
    }


def _increments_from_poses(poses):
    return np.asarray(
        [_relative_np(poses[index], poses[index + 1]) for index in range(len(poses) - 1)],
        dtype=np.float64,
    )


def _pair_summary(increments, rows):
    target = np.asarray(
        [
            [row["target_forward_m"], row["target_left_m"], row["target_yaw_rad"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    trans = np.linalg.norm(increments[:, :2] - target[:, :2], axis=1)
    yaw = np.abs([_wrap_np(float(a - b)) for a, b in zip(increments[:, 2], target[:, 2])])
    pred_dist = float(np.linalg.norm(increments[:, :2], axis=1).sum())
    target_dist = float(np.linalg.norm(target[:, :2], axis=1).sum())
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(trans * trans))),
        "yaw_rmse_deg": math.degrees(float(np.sqrt(np.mean(yaw * yaw)))),
        "predicted_pair_distance_sum_m": pred_dist,
        "target_pair_distance_sum_m": target_dist,
        "pair_distance_ratio": pred_dist / target_dist if target_dist > 1e-9 else None,
    }


def _optimize_sequence(
    rows,
    add_closed_loop=False,
    loop_weight=25.0,
    odom_translation_weight=1.0,
    odom_yaw_weight=1.0,
    iterations=200,
):
    initial, target_poses = _integrate(rows)
    measurements = np.asarray(
        [
            [row["pred_forward_m"], row["pred_left_m"], row["pred_yaw_rad"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    device = torch.device("cpu")
    variable = torch.tensor(initial[1:], dtype=torch.float64, device=device, requires_grad=True)
    first = torch.zeros((1, 3), dtype=torch.float64, device=device)
    measured = torch.tensor(measurements, dtype=torch.float64, device=device)
    odom_weights = torch.tensor(
        [odom_translation_weight, odom_translation_weight, odom_yaw_weight],
        dtype=torch.float64,
        device=device,
    )
    optimizer = torch.optim.LBFGS(
        [variable],
        max_iter=int(iterations),
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        poses = torch.cat((first, variable), dim=0)
        predicted_relative = _relative_torch(poses[:-1], poses[1:])
        residual_raw = predicted_relative - measured
        residual = torch.stack(
            (
                residual_raw[:, 0],
                residual_raw[:, 1],
                _wrap_torch(residual_raw[:, 2]),
            ),
            dim=1,
        )
        loss = torch.mean((residual * odom_weights) ** 2)
        if add_closed_loop:
            loop_relative = _relative_torch(poses[-1], poses[0])
            loop_residual = loop_relative
            loop_residual = torch.stack(
                (
                    loop_residual[0],
                    loop_residual[1],
                    _wrap_torch(loop_residual[2]),
                )
            )
            loop_weights = torch.tensor(
                [loop_weight, loop_weight, loop_weight],
                dtype=torch.float64,
                device=device,
            )
            loss = loss + torch.mean((loop_residual * loop_weights) ** 2)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        optimized = torch.cat((first, variable), dim=0).detach().cpu().numpy()
    optimized[:, 2] = np.asarray([_wrap_np(value) for value in optimized[:, 2]])
    return initial, optimized, target_poses


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--loop-weight", type=float, default=25.0)
    parser.add_argument("--odom-translation-weight", type=float, default=1.0)
    parser.add_argument("--odom-yaw-weight", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=200)
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    rows = _read_predictions(args.predictions)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    result = {
        "predictions": str(Path(args.predictions).resolve()),
        "closed_loop": args.closed_loop,
        "loop_weight": args.loop_weight,
        "odom_translation_weight": args.odom_translation_weight,
        "odom_yaw_weight": args.odom_yaw_weight,
        "iterations": args.iterations,
        "sequences": {},
    }
    for sequence, sequence_rows in grouped.items():
        sequence_rows = _main_chain(sequence_rows)
        raw, optimized, target = _optimize_sequence(
            sequence_rows,
            add_closed_loop=args.closed_loop,
            loop_weight=args.loop_weight,
            odom_translation_weight=args.odom_translation_weight,
            odom_yaw_weight=args.odom_yaw_weight,
            iterations=args.iterations,
        )
        result["sequences"][sequence] = {
            "chain_pairs": len(sequence_rows),
            "raw_trajectory": _trajectory_summary(raw, target),
            "optimized_trajectory": _trajectory_summary(optimized, target),
            "raw_pair": _pair_summary(_increments_from_poses(raw), sequence_rows),
            "optimized_pair": _pair_summary(
                _increments_from_poses(optimized), sequence_rows
            ),
        }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
