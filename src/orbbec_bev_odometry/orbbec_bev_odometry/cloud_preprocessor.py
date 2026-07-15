"""Filtering and deterministic voxel sampling for Orbbec depth clouds."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PreprocessConfig:
    voxel_size: float = 0.06
    min_range: float = 0.35
    max_range: float = 6.0
    # The camera is low on the recorded platform; this removes the dominant floor.
    min_height: float = -0.15
    max_height: float = 1.2
    outlier_neighbors: int = 0
    outlier_std_ratio: float = 2.5
    max_points: int = 9000


class CloudPreprocessor:
    """Convert optical XYZ into base planar coordinates and reject noise."""

    def __init__(self, config=None):
        self.config = config or PreprocessConfig()

    @staticmethod
    def voxel_downsample(points, voxel_size):
        if len(points) == 0:
            return points
        keys = np.floor(points / voxel_size).astype(np.int32)
        _, first = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(first)]

    def process(self, xyz):
        """Return XYZ as [forward, left, up] metres after all filters."""
        xyz = np.asarray(xyz, dtype=np.float64)
        valid = np.isfinite(xyz).all(axis=1)
        optical = xyz[valid]
        # Orbbec optical axes: +x right, +y down, +z forward.
        points = np.column_stack((optical[:, 2], -optical[:, 0], -optical[:, 1]))
        planar_range = np.linalg.norm(points[:, :2], axis=1)
        cfg = self.config
        keep = (
            (planar_range >= cfg.min_range)
            & (planar_range <= cfg.max_range)
            & (points[:, 2] >= cfg.min_height)
            & (points[:, 2] <= cfg.max_height)
        )
        points = self.voxel_downsample(points[keep], cfg.voxel_size)
        if cfg.outlier_neighbors > 0 and len(points) > cfg.outlier_neighbors + 1:
            distances, _ = cKDTree(points).query(
                points, k=cfg.outlier_neighbors + 1
            )
            mean_distance = distances[:, 1:].mean(axis=1)
            threshold = np.median(mean_distance) + cfg.outlier_std_ratio * 1.4826 * np.median(
                np.abs(mean_distance - np.median(mean_distance))
            )
            points = points[mean_distance <= threshold]
        if len(points) > cfg.max_points:
            indices = np.linspace(0, len(points) - 1, cfg.max_points).astype(int)
            points = points[indices]
        return points
