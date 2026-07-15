import numpy as np

from orbbec_bev_odometry.planar_icp import IcpConfig, PlanarPointToPlaneICP
from orbbec_bev_odometry.se2 import inverse, matrix, transform_points


def test_recovers_planar_transform():
    wall_x = np.column_stack((np.full(250, 2.0), np.linspace(-2.0, 2.0, 250)))
    wall_y = np.column_stack((np.linspace(-2.0, 2.0, 250), np.full(250, 1.5)))
    target = np.vstack((wall_x, wall_y))
    expected = matrix(0.08, -0.05, 0.035)
    source = transform_points(inverse(expected), target)
    result = PlanarPointToPlaneICP(IcpConfig(min_correspondences=80)).register(source, target)
    assert result.accepted
    assert np.allclose(result.transform, expected, atol=2e-3)
