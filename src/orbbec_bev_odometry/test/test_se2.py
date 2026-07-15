import numpy as np

from orbbec_bev_odometry.se2 import inverse, matrix, transform_points


def test_inverse_round_trip():
    pose = matrix(1.2, -0.4, 0.7)
    points = np.array([[0.0, 0.0], [1.0, 2.0]])
    assert np.allclose(transform_points(inverse(pose), transform_points(pose, points)), points)
