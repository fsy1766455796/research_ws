import math

import numpy as np

from orbbec_bev_odometry.openloris_geometry import (
    interpolate_pose,
    relative_planar_pose,
)


def test_groundtruth_interpolation_and_relative_pose():
    half_yaw = math.radians(10.0) * 0.5
    groundtruth = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 0.0, 0.0, 0.0,
             math.sin(half_yaw), math.cos(half_yaw)],
        ]
    )
    pose_0 = interpolate_pose(groundtruth, 0.25, max_gap_sec=1.0)
    pose_1 = interpolate_pose(groundtruth, 0.75, max_gap_sec=1.0)
    relative = relative_planar_pose(pose_0, pose_1)

    assert np.isclose(relative[0], 1.0, atol=0.05)
    assert np.isclose(math.degrees(relative[2]), 5.0, atol=0.01)
