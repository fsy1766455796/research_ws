"""Rolling scan-to-local-submap storage."""

from collections import deque

import numpy as np

from .cloud_preprocessor import CloudPreprocessor
from .se2 import inverse, transform_points


class LocalSubmap:
    def __init__(self, max_scans=12, voxel_size=0.08, max_points=16000):
        self.scans = deque(maxlen=max_scans)
        self.voxel_size = voxel_size
        self.max_points = max_points

    def add(self, planar_points, world_pose):
        self.scans.append((np.array(planar_points, copy=True), np.array(world_pose, copy=True)))

    def target_in_frame(self, predicted_world_pose):
        """Build the rolling world submap expressed in the predicted scan frame."""
        if not self.scans:
            return np.empty((0, 2))
        world = np.concatenate([transform_points(pose, points) for points, pose in self.scans])
        local = transform_points(inverse(predicted_world_pose), world)
        local = CloudPreprocessor.voxel_downsample(local, self.voxel_size)
        if len(local) > self.max_points:
            indices = np.linspace(0, len(local) - 1, self.max_points).astype(int)
            local = local[indices]
        return local
