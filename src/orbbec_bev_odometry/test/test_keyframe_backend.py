import math

import numpy as np

from orbbec_bev_odometry.keyframe_backend import KeyframeConfig, run_keyframe_backend
from orbbec_bev_odometry.local_map_backend import compose, metrics, relative
from orbbec_bev_odometry.se2 import inverse, matrix, transform_points


def _world():
    walls = []
    for (x0, y0), (x1, y1) in [((-2, -3), (6, -3)), ((6, -3), (6, 4)), ((6, 4), (-2, 4)),
                               ((-2, 4), (-2, -3)), ((1, 0.5), (2.5, 0.5)), ((2.5, 0.5), (2.5, 2)),
                               ((3.5, -1.5), (4.5, -2.2)), ((0, -1.8), (0.8, -1.2))]:
        n = int(math.hypot(x1 - x0, y1 - y0) / 0.03)
        walls.append(np.column_stack((np.linspace(x0, x1, n), np.linspace(y0, y1, n))))
    return np.concatenate(walls)


def _scan(world, pose):
    local = transform_points(inverse(matrix(*pose)), world)
    rng = np.hypot(local[:, 0], local[:, 1])
    bearing = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
    keep = (rng > 0.3) & (rng < 5.0) & (np.abs(bearing) < 50)
    return local[keep]


def _trajectory(count=150):
    poses = [np.zeros(3)]
    for i in range(count):
        yaw_rate = math.radians(1.2) if 40 <= i < 110 else 0.0
        poses.append(compose(poses[-1], [0.03, 0.0, yaw_rate]))
    return np.asarray(poses)


def _increments(gt, yaw_bias_deg=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for a, b in zip(gt[:-1], gt[1:]):
        inc = relative(a, b)
        inc[:2] += rng.normal(0, 0.003, 2)
        inc[2] += math.radians(yaw_bias_deg) + rng.normal(0, math.radians(0.1))
        out.append(inc)
    return np.asarray(out)


def test_backend_removes_most_yaw_drift():
    world, gt = _world(), _trajectory()
    increments = _increments(gt, yaw_bias_deg=0.4)
    raw, optimized, keyframes, diagnostics = run_keyframe_backend(
        increments, lambda i: _scan(world, gt[i]), KeyframeConfig())
    before, after = metrics(raw, gt), metrics(optimized, gt)
    assert before['endpoint_yaw_error_deg'] > 50
    assert after['endpoint_yaw_error_deg'] < 0.3 * before['endpoint_yaw_error_deg']
    assert after['position_rmse_start_aligned_m'] < 0.5 * before['position_rmse_start_aligned_m']
    assert len(keyframes) > 10 and sum(d['neighbor_edges'] for d in diagnostics) > 20


def test_backend_does_not_degrade_good_odometry():
    world, gt = _world(), _trajectory()
    increments = _increments(gt, yaw_bias_deg=0.0, seed=1)
    raw, optimized, _, _ = run_keyframe_backend(
        increments, lambda i: _scan(world, gt[i]), KeyframeConfig())
    assert metrics(optimized, gt)['position_rmse_start_aligned_m'] < 0.1
    assert metrics(optimized, gt)['endpoint_yaw_error_deg'] < 2.0
