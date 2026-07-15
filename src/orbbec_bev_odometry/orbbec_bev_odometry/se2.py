"""Small, dependency-light SE(2) helpers used by point-cloud odometry."""

import math

import numpy as np


def matrix(x=0.0, y=0.0, yaw=0.0):
    """Return a homogeneous transform mapping local planar points to world."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def components(transform):
    """Return x, y and wrapped yaw from a homogeneous transform."""
    return (
        float(transform[0, 2]),
        float(transform[1, 2]),
        math.atan2(transform[1, 0], transform[0, 0]),
    )


def inverse(transform):
    """Invert an SE(2) transform."""
    rotation = transform[:2, :2]
    result = np.eye(3)
    result[:2, :2] = rotation.T
    result[:2, 2] = -rotation.T @ transform[:2, 2]
    return result


def transform_points(transform, points):
    """Transform an N-by-2 point array."""
    return points @ transform[:2, :2].T + transform[:2, 2]


def wrap_angle(angle):
    """Wrap radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
