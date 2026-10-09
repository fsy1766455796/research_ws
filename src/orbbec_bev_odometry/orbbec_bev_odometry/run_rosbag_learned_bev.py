"""Run learned BEV odometry on a rosbag2 MCAP depth recording."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from .learned_bev_model import LearnedBEVOdometry
from .multichannel_bev import MultiChannelBEVConfig, depth_to_multichannel_bev


def _load_checkpoint(filename, device):
    try:
        return torch.load(filename, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=device)


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _compose(pose, increment, yaw_gain=1.0):
    x, y, yaw = pose
    forward, left, dyaw = increment
    dyaw *= float(yaw_gain)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            x + cosine * forward - sine * left,
            y + sine * forward + cosine * left,
            _wrap(yaw + dyaw),
        ],
        dtype=np.float64,
    )


def _apply_motion_gate(
    prediction,
    static_step_deadband_m=0.0,
    static_yaw_deadband_deg=0.0,
):
    gated = prediction.copy()
    step = math.hypot(float(gated[0]), float(gated[1]))
    yaw_abs_deg = abs(math.degrees(float(gated[2])))
    if (
        step < float(static_step_deadband_m)
        and yaw_abs_deg < float(static_yaw_deadband_deg)
    ):
        gated[:] = 0.0
    return gated


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
            torch.atan2(
                torch.sin(pose_j[..., 2] - pose_i[..., 2]),
                torch.cos(pose_j[..., 2] - pose_i[..., 2]),
            ),
        ),
        dim=-1,
    )


def _optimize_closed_loop(
    poses,
    measurements,
    loop_weight=5.0,
    odom_translation_weight=1.0,
    odom_yaw_weight=1.0,
    iterations=200,
):
    device = torch.device("cpu")
    first = torch.zeros((1, 3), dtype=torch.float64, device=device)
    variable = torch.tensor(
        poses[1:], dtype=torch.float64, device=device, requires_grad=True
    )
    measured = torch.tensor(measurements, dtype=torch.float64, device=device)
    odom_weights = torch.tensor(
        [odom_translation_weight, odom_translation_weight, odom_yaw_weight],
        dtype=torch.float64,
        device=device,
    )
    loop_weights = torch.tensor(
        [loop_weight, loop_weight, loop_weight],
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
        graph_poses = torch.cat((first, variable), dim=0)
        predicted = _relative_torch(graph_poses[:-1], graph_poses[1:])
        residual_raw = predicted - measured
        residual = torch.stack(
            (
                residual_raw[:, 0],
                residual_raw[:, 1],
                torch.atan2(torch.sin(residual_raw[:, 2]), torch.cos(residual_raw[:, 2])),
            ),
            dim=1,
        )
        odom_loss = torch.mean((residual * odom_weights) ** 2)
        loop_relative = _relative_torch(graph_poses[-1], graph_poses[0])
        loop_residual = torch.stack(
            (
                loop_relative[0],
                loop_relative[1],
                torch.atan2(torch.sin(loop_relative[2]), torch.cos(loop_relative[2])),
            )
        )
        loop_loss = torch.mean((loop_residual * loop_weights) ** 2)
        loss = odom_loss + loop_loss
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        optimized = torch.cat((first, variable), dim=0).detach().cpu().numpy()
    optimized[:, 2] = np.asarray([_wrap(value) for value in optimized[:, 2]])
    return optimized


def _rotation_x(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _camera_transform(camera_z, camera_pitch_deg=0.0, camera_yaw_deg=0.0):
    optical_to_base = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, float(camera_z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adjustment = np.eye(4, dtype=np.float64)
    adjustment[:3, :3] = (
        _rotation_z(math.radians(float(camera_yaw_deg)))
        @ _rotation_x(math.radians(float(camera_pitch_deg)))
    )
    return adjustment @ optical_to_base


def _depth_from_image_msg(msg):
    if msg.encoding != "16UC1":
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) * 0.001


def _load_depth_frames(bag_path, depth_topic, camera_info_topic, frame_stride, max_frames):
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError("Install rosbags to read rosbag2 MCAP files") from error

    frames = []
    intrinsics = None
    bag = Path(bag_path)
    with AnyReader([bag]) as reader:
        connections = [
            connection for connection in reader.connections
            if connection.topic in (depth_topic, camera_info_topic)
        ]
        depth_count = 0
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.topic == camera_info_topic and intrinsics is None:
                intrinsics = (
                    float(msg.k[0]),
                    float(msg.k[4]),
                    float(msg.k[2]),
                    float(msg.k[5]),
                )
            elif connection.topic == depth_topic:
                if depth_count % int(frame_stride) == 0:
                    depth = _depth_from_image_msg(msg)
                    frames.append((timestamp, depth))
                    if max_frames and len(frames) >= int(max_frames):
                        break
                depth_count += 1
    if intrinsics is None:
        raise ValueError(f"No CameraInfo found on topic {camera_info_topic}")
    if len(frames) < 2:
        raise ValueError("Need at least two depth frames")
    return intrinsics, frames


def _run_frontend(model, previous_bev, current_bev, device):
    previous = torch.from_numpy(previous_bev).unsqueeze(0).to(device)
    current = torch.from_numpy(current_bev).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(previous, current)
    pose = output["pose"][0].detach().cpu().numpy().astype(np.float64)
    confidence = float(output["confidence"][0].detach().cpu())
    return pose, confidence


def _trajectory_summary(poses):
    increments = np.asarray([
        math.hypot(poses[index + 1, 0] - poses[index, 0],
                   poses[index + 1, 1] - poses[index, 1])
        for index in range(len(poses) - 1)
    ])
    endpoint = poses[-1]
    return {
        "node_count": int(len(poses)),
        "path_chord_m": float(np.linalg.norm(endpoint[:2])),
        "endpoint_x_m": float(endpoint[0]),
        "endpoint_y_m": float(endpoint[1]),
        "endpoint_yaw_deg": math.degrees(float(endpoint[2])),
        "step_distance_sum_m": float(increments.sum()),
    }


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/depth/camera_info")
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--camera-z", type=float, default=0.8)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    parser.add_argument("--yaw-gain", type=float, default=1.0)
    parser.add_argument("--static-step-deadband-m", type=float, default=0.0)
    parser.add_argument("--static-yaw-deadband-deg", type=float, default=0.0)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--loop-weight", type=float, default=5.0)
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
    bev_config_values = dict(checkpoint.get("bev_config") or {})
    if "height_bins_m" in bev_config_values:
        bev_config_values["height_bins_m"] = tuple(bev_config_values["height_bins_m"])
    bev_config = MultiChannelBEVConfig(**bev_config_values)

    intrinsics, depth_frames = _load_depth_frames(
        args.bag,
        args.depth_topic,
        args.camera_info_topic,
        args.frame_stride,
        args.max_frames,
    )
    transform = _camera_transform(
        args.camera_z,
        camera_pitch_deg=args.camera_pitch_deg,
        camera_yaw_deg=args.camera_yaw_deg,
    )
    bevs = []
    for _, depth in depth_frames:
        bevs.append(
            depth_to_multichannel_bev(
                depth,
                config=bev_config,
                intrinsics=intrinsics,
                t_base_camera=transform,
                pixel_stride=args.pixel_stride,
            )
        )

    rows = []
    poses = [np.zeros(3, dtype=np.float64)]
    measurements = []
    gated_count = 0
    for index in range(len(bevs) - 1):
        prediction, confidence = _run_frontend(model, bevs[index], bevs[index + 1], device)
        gated_prediction = _apply_motion_gate(
            prediction,
            static_step_deadband_m=args.static_step_deadband_m,
            static_yaw_deadband_deg=args.static_yaw_deadband_deg,
        )
        if not np.any(gated_prediction):
            gated_count += 1
        calibrated_prediction = gated_prediction.copy()
        calibrated_prediction[2] *= float(args.yaw_gain)
        measurements.append(calibrated_prediction)
        poses.append(_compose(poses[-1], gated_prediction, yaw_gain=args.yaw_gain))
        rows.append(
            {
                "index": index,
                "timestamp_0": int(depth_frames[index][0]),
                "timestamp_1": int(depth_frames[index + 1][0]),
                "pred_forward_m": float(prediction[0]),
                "pred_left_m": float(prediction[1]),
                "pred_yaw_rad": float(prediction[2]),
                "gated_forward_m": float(gated_prediction[0]),
                "gated_left_m": float(gated_prediction[1]),
                "gated_yaw_rad": float(gated_prediction[2]),
                "calibrated_yaw_rad": float(calibrated_prediction[2]),
                "confidence": confidence,
                "pose_x_m": float(poses[-1][0]),
                "pose_y_m": float(poses[-1][1]),
                "pose_yaw_rad": float(poses[-1][2]),
            }
        )
    poses = np.asarray(poses, dtype=np.float64)
    measurements = np.asarray(measurements, dtype=np.float64)
    optimized = None
    if args.closed_loop:
        optimized = _optimize_closed_loop(
            poses,
            measurements,
            loop_weight=args.loop_weight,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savetxt(output_dir / "raw_trajectory.csv", poses, delimiter=",", header="x_m,y_m,yaw_rad", comments="")
    if optimized is not None:
        np.savetxt(output_dir / "optimized_trajectory.csv", optimized, delimiter=",", header="x_m,y_m,yaw_rad", comments="")
    summary = {
        "bag": str(Path(args.bag).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "frames_used": len(depth_frames),
        "frame_stride": args.frame_stride,
        "camera_z": args.camera_z,
        "camera_pitch_deg": args.camera_pitch_deg,
        "camera_yaw_deg": args.camera_yaw_deg,
        "yaw_gain": args.yaw_gain,
        "static_step_deadband_m": args.static_step_deadband_m,
        "static_yaw_deadband_deg": args.static_yaw_deadband_deg,
        "gated_pair_count": gated_count,
        "intrinsics_fx_fy_cx_cy": intrinsics,
        "closed_loop": bool(args.closed_loop),
        "raw": _trajectory_summary(poses),
        "optimized": _trajectory_summary(optimized) if optimized is not None else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
