import numpy as np

from orbbec_bev_odometry.bev_odometry_node import ground_plane_mask


def test_ground_plane_is_removed_but_obstacle_is_retained():
    rng = np.random.default_rng(7)

    ground_x = rng.uniform(-2000.0, 2000.0, 1500)
    ground_z = rng.uniform(300.0, 5500.0, 1500)
    ground_y = 430.0 + 0.03 * ground_z + rng.normal(0.0, 6.0, 1500)
    ground = np.column_stack((ground_x, ground_y, ground_z)).astype(np.float32)

    obstacle_x = rng.uniform(-250.0, 250.0, 300)
    obstacle_y = rng.uniform(-500.0, 250.0, 300)
    obstacle_z = rng.uniform(1800.0, 2200.0, 300)
    obstacle = np.column_stack((obstacle_x, obstacle_y, obstacle_z)).astype(np.float32)

    points = np.vstack((ground, obstacle))
    mask = ground_plane_mask(
        points,
        distance_threshold_mm=40.0,
        max_tilt_deg=25.0,
        ransac_iterations=40,
        sample_size=5000,
        min_inliers=150,
        min_inlier_ratio=0.08,
        candidate_min_y_mm=100.0,
    )

    assert np.mean(mask[: ground.shape[0]]) > 0.98
    assert np.mean(mask[ground.shape[0] :]) < 0.05


def test_no_confident_plane_keeps_all_points():
    rng = np.random.default_rng(11)
    points = rng.uniform(
        (-500.0, -500.0, 500.0),
        (500.0, 50.0, 2000.0),
        (100, 3),
    )

    mask = ground_plane_mask(
        points.astype(np.float32),
        distance_threshold_mm=40.0,
        max_tilt_deg=25.0,
        ransac_iterations=20,
        sample_size=5000,
        min_inliers=150,
        min_inlier_ratio=0.08,
        candidate_min_y_mm=100.0,
    )

    assert not np.any(mask)
