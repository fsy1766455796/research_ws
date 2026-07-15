"""Run point-cloud odometry directly on a rosbag and compare recorded BEV odom."""

import argparse
import json
import math
import statistics

import numpy as np
import rosbag2_py
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2

from .pointcloud_odometry_node import xyz_view
from .pointcloud_odom_core import PointCloudOdomCore
from .se2 import components


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def trajectory_metrics(samples):
    if not samples:
        return {}
    start, end = samples[0], samples[-1]
    closure_position = math.hypot(end[1] - start[1], end[2] - start[2])
    closure_yaw = abs(math.degrees((end[3] - start[3] + math.pi) % (2 * math.pi) - math.pi))
    duration = end[0] - start[0]
    stationary = [s for s in samples if s[0] - start[0] <= min(10.0, duration * 0.1)]
    sx, sy = stationary[0][1], stationary[0][2]
    stationary_drift = max((math.hypot(s[1] - sx, s[2] - sy) for s in stationary), default=0.0)
    return {"frames": len(samples), "closure_yaw_deg": closure_yaw,
            "closure_position_m": closure_position, "stationary_drift_m": stationary_drift}


def run(bag, interval, output):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    core, pointcloud, bev, timings = PointCloudOdomCore(), [], [], []
    first_ns = last_cloud_ns = None
    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if first_ns is None:
            first_ns = timestamp
        if topic == "/bev_odom":
            msg = deserialize_message(data, Odometry); pose = msg.pose.pose
            bev.append(((timestamp - first_ns) * 1e-9, pose.position.x, pose.position.y,
                        yaw_from_quaternion(pose.orientation)))
        elif topic == "/camera/depth/points" and (
                last_cloud_ns is None or timestamp - last_cloud_ns >= interval * 1e9):
            last_cloud_ns = timestamp
            result = core.process(xyz_view(deserialize_message(data, PointCloud2)))
            x, y, yaw = components(result.pose)
            pointcloud.append(((timestamp - first_ns) * 1e-9, x, y, yaw))
            timings.append(result.processing_ms)
            if len(pointcloud) % 50 == 0:
                print(f"processed={len(pointcloud)} pose=({x:+.3f},{y:+.3f},{math.degrees(yaw):+.2f}deg) "
                      f"last={result.processing_ms:.1f}ms", flush=True)
    report = {"bag": bag, "interval_sec": interval, "bev_ecc": trajectory_metrics(bev),
              "pointcloud_icp": trajectory_metrics(pointcloud)}
    if timings:
        report["pointcloud_icp"].update({"processing_mean_ms": statistics.mean(timings),
            "processing_p95_ms": float(np.quantile(timings, 0.95)),
            "processing_max_ms": max(timings), "realtime_factor": interval * 1000 / statistics.mean(timings)})
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if output:
        with open(output, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag"); parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output"); args = parser.parse_args()
    run(args.bag, args.interval, args.output)
