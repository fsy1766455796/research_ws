"""Timestamp and pose helpers for OpenLORIS-Scene training data."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class TimestampedPath:
    """Associate a sensor timestamp with its relative data path."""

    timestamp: float
    path: Path


def load_timestamped_paths(index_file):
    """Read a TUM-style ``timestamp relative/path`` index file."""
    index_path = Path(index_file)
    entries: List[TimestampedPath] = []
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(maxsplit=1)
            if len(fields) != 2:
                continue
            try:
                timestamp = float(fields[0])
            except ValueError:
                continue
            entries.append(TimestampedPath(timestamp, Path(fields[1])))
    return entries


def load_groundtruth(filename):
    """Return sorted ``[timestamp, px, py, pz, qx, qy, qz, qw]`` rows."""
    rows = []
    with Path(filename).open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.strip().split()
            if len(fields) != 8:
                continue
            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No ground-truth poses found in {filename}")
    result = np.asarray(rows, dtype=np.float64)
    return result[np.argsort(result[:, 0])]


def quaternion_to_rotation(quaternion):
    """Convert an ``[x, y, z, w]`` quaternion into a rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm((x, y, z, w))
    if norm < 1e-12:
        raise ValueError("Quaternion has zero length")
    x, y, z, w = (x / norm, y / norm, z / norm, w / norm)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
             2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w),
             1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(q0, q1, fraction):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + float(fraction) * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - fraction) * theta) / sin_theta * q0
        + np.sin(fraction * theta) / sin_theta * q1
    )


def _pose_from_row(row):
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_to_rotation(row[4:8])
    pose[:3, 3] = row[1:4]
    return pose


def interpolate_pose(groundtruth, timestamp, max_gap_sec=0.05):
    """Interpolate a base_link pose at a sensor timestamp.

    OpenLORIS package ground truth is already expressed for ``base_link``.
    ``None`` is returned when the timestamp is outside the trajectory or the
    surrounding measurements are too far away.
    """
    gt = np.asarray(groundtruth, dtype=np.float64)
    if abs(float(timestamp) - gt[0, 0]) <= 1e-9:
        return _pose_from_row(gt[0])
    if abs(float(timestamp) - gt[-1, 0]) <= 1e-9:
        return _pose_from_row(gt[-1])
    index = int(np.searchsorted(gt[:, 0], float(timestamp)))
    if index == 0 or index >= len(gt):
        return None
    before = gt[index - 1]
    after = gt[index]
    if abs(float(timestamp) - after[0]) <= 1e-9:
        return _pose_from_row(after)
    if max(timestamp - before[0], after[0] - timestamp) > max_gap_sec:
        return None
    duration = after[0] - before[0]
    fraction = 0.0 if duration <= 0.0 else (timestamp - before[0]) / duration
    translation = before[1:4] + fraction * (after[1:4] - before[1:4])
    quaternion = _slerp(before[4:8], after[4:8], fraction)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quaternion_to_rotation(quaternion)
    pose[:3, 3] = translation
    return pose


def relative_planar_pose(world_from_base_0, world_from_base_1):
    """Return base-1 pose in base-0 as ``[forward, left, yaw]``."""
    relative = np.linalg.inv(world_from_base_0) @ world_from_base_1
    yaw = np.arctan2(relative[1, 0], relative[0, 0])
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    return np.array(
        [relative[0, 3], relative[1, 3], yaw], dtype=np.float32
    )
