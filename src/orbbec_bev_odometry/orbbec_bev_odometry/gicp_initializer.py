"""Thin, testable wrapper around the optional native PCL GICP initializer."""

import numpy as np

from .planar_icp import IcpResult
from .se2 import matrix

try:
    from . import _pcl_gicp
except ImportError:  # Source-tree unit tests can run before the extension is built.
    _pcl_gicp = None


class GicpInitializer:
    def __init__(self, max_points=5000, max_correspondence=0.45, iterations=30):
        self.max_points = max_points
        self.max_correspondence = max_correspondence
        self.iterations = iterations

    def register(self, source, target):
        if _pcl_gicp is None or len(source) < 100 or len(target) < 100:
            return None
        def sample(points):
            if len(points) <= self.max_points:
                return np.ascontiguousarray(points, dtype=np.float64)
            index = np.linspace(0, len(points) - 1, self.max_points).astype(int)
            return np.ascontiguousarray(points[index], dtype=np.float64)
        x, y, yaw, fitness, converged = _pcl_gicp.align(
            sample(source), sample(target), self.max_correspondence, self.iterations
        )
        transform = matrix(x, y, yaw)
        accepted = bool(converged and fitness < 0.04 and np.hypot(x, y) < 0.60
                        and abs(yaw) < 0.60)
        return IcpResult(transform, accepted, float(np.sqrt(max(fitness, 0.0))),
                         0.0, min(len(source), len(target)), self.iterations, 1.0)
