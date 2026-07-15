"""Robust point-to-plane ICP constrained directly to planar SE(2)."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree

from .se2 import components, matrix, transform_points


@dataclass
class IcpConfig:
    max_iterations: int = 18
    max_correspondence: float = 0.35
    normal_neighbors: int = 12
    min_correspondences: int = 180
    trim_fraction: float = 0.82
    huber_delta: float = 0.045
    convergence_translation: float = 0.0005
    convergence_yaw: float = 0.0002
    max_step_translation: float = 0.45
    max_step_yaw: float = 0.35
    coarse_yaw_range: float = 0.50
    coarse_translation_range: float = 0.45
    coarse_max_points: int = 1200


@dataclass
class IcpResult:
    transform: np.ndarray
    accepted: bool
    rmse: float
    inlier_ratio: float
    correspondences: int
    iterations: int
    condition: float


class PlanarPointToPlaneICP:
    """Register source scan into a target submap while optimizing x/y/yaw."""

    def __init__(self, config=None):
        self.config = config or IcpConfig()

    def estimate_normals(self, target):
        tree = cKDTree(target)
        _, indices = tree.query(target, k=min(self.config.normal_neighbors, len(target)))
        neighborhoods = target[indices]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        values, vectors = np.linalg.eigh(covariance)
        normals = vectors[:, :, 0]
        curvature = values[:, 0] / np.maximum(values.sum(axis=1), 1e-12)
        return tree, normals, curvature

    def coarse_initial(self, source, target, tree):
        """Find a basin for ICP during fast turns using robust NN scoring."""
        cfg = self.config
        stride = max(1, len(source) // cfg.coarse_max_points)
        sample = source[::stride]

        def score(parameters):
            moved = transform_points(matrix(*parameters), sample)
            distances, _ = tree.query(moved)
            return float(np.quantile(distances, 0.55))

        xy = cfg.coarse_translation_range
        result = differential_evolution(
            score, [(-xy, xy), (-xy, xy), (-cfg.coarse_yaw_range, cfg.coarse_yaw_range)],
            maxiter=12, popsize=7, tol=0.03, polish=True, seed=7, workers=1,
        )
        return matrix(*result.x)

    def register_point_to_point(self, source, target, initial=None):
        """Robust SE(2) ICP used only to seed scan-to-submap refinement."""
        cfg = self.config
        transform = np.eye(3) if initial is None else np.array(initial, copy=True)
        if len(source) < cfg.min_correspondences or len(target) < cfg.min_correspondences:
            return IcpResult(transform, False, np.inf, 0.0, 0, 0, np.inf)
        tree = cKDTree(target)
        count, rmse = 0, np.inf
        for iteration in range(cfg.max_iterations):
            moved = transform_points(transform, source)
            distances, indices = tree.query(moved)
            selected = np.flatnonzero(distances < cfg.max_correspondence)
            if len(selected) < cfg.min_correspondences:
                break
            cutoff = np.quantile(distances[selected], cfg.trim_fraction)
            selected = selected[distances[selected] <= cutoff]
            a, b = moved[selected], target[indices[selected]]
            ac, bc = a.mean(axis=0), b.mean(axis=0)
            u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0:
                vt[-1] *= -1
                rotation = vt.T @ u.T
            increment = np.eye(3)
            increment[:2, :2] = rotation
            increment[:2, 2] = bc - rotation @ ac
            transform = increment @ transform
            count = len(selected)
            rmse = float(np.sqrt(np.mean(distances[selected] ** 2)))
            _, _, dyaw = components(increment)
            if np.linalg.norm(increment[:2, 2]) < cfg.convergence_translation and abs(dyaw) < cfg.convergence_yaw:
                break
        x, y, yaw = components(transform)
        accepted = (count >= cfg.min_correspondences and rmse < 0.12
                    and np.hypot(x, y) <= cfg.max_step_translation
                    and abs(yaw) <= cfg.max_step_yaw)
        return IcpResult(transform, accepted, rmse, count / max(len(source), 1),
                         count, iteration + 1, 1.0)

    @staticmethod
    def _voxel_2d(points, size):
        keys = np.floor(points / size).astype(np.int32)
        _, first = np.unique(keys, axis=0, return_index=True)
        return points[np.sort(first)]

    def register_multiscale(self, source, target, initial=None):
        """Coarse-to-fine point-to-plane ICP for adjacent-frame initialization."""
        transform = np.eye(3) if initial is None else np.array(initial, copy=True)
        final = IcpResult(transform, False, np.inf, 0.0, 0, 0, np.inf)
        # Large basins recover fast turns; the final level restores centimetre accuracy.
        for voxel, distance, iterations in ((0.18, 0.65, 12), (0.10, 0.40, 14),
                                             (0.06, 0.24, 18)):
            level_source = self._voxel_2d(source, voxel)
            level_target = self._voxel_2d(target, voxel)
            minimum = min(120, max(35, min(len(level_source), len(level_target)) // 8))
            solver = PlanarPointToPlaneICP(replace(
                self.config, max_correspondence=distance,
                max_iterations=iterations, min_correspondences=minimum,
                max_step_translation=0.60, max_step_yaw=0.55,
            ))
            final = solver.register(level_source, level_target, initial=transform)
            transform = final.transform
            if final.correspondences < minimum:
                break
        x, y, yaw = components(transform)
        accepted = (final.correspondences >= 35 and final.rmse < 0.10
                    and np.hypot(x, y) <= 0.60 and abs(yaw) <= 0.55)
        return IcpResult(transform, accepted, final.rmse, final.inlier_ratio,
                         final.correspondences, final.iterations, final.condition)

    def register(self, source, target, initial=None):
        cfg = self.config
        transform = np.eye(3) if initial is None else np.array(initial, copy=True)
        if len(source) < cfg.min_correspondences or len(target) < cfg.min_correspondences:
            return IcpResult(transform, False, np.inf, 0.0, 0, 0, np.inf)
        tree, normals, curvature = self.estimate_normals(target)
        if initial is None:
            transform = self.coarse_initial(source, target, tree)
        count, condition, rmse = 0, np.inf, np.inf
        for iteration in range(cfg.max_iterations):
            moved = transform_points(transform, source)
            distances, indices = tree.query(moved)
            mask = (distances < cfg.max_correspondence) & (curvature[indices] < 0.12)
            selected = np.flatnonzero(mask)
            if len(selected) < cfg.min_correspondences:
                break
            residual = np.einsum(
                "ij,ij->i", normals[indices[selected]], moved[selected] - target[indices[selected]]
            )
            trim = np.quantile(np.abs(residual), cfg.trim_fraction)
            selected = selected[np.abs(residual) <= trim]
            residual = np.einsum(
                "ij,ij->i", normals[indices[selected]], moved[selected] - target[indices[selected]]
            )
            n = normals[indices[selected]]
            p = moved[selected]
            jacobian = np.column_stack((n[:, 0], n[:, 1], n[:, 1] * p[:, 0] - n[:, 0] * p[:, 1]))
            weights = np.minimum(1.0, cfg.huber_delta / np.maximum(np.abs(residual), 1e-12))
            normal_matrix = jacobian.T @ (weights[:, None] * jacobian)
            rhs = -(jacobian.T @ (weights * residual))
            condition = np.linalg.cond(normal_matrix)
            if not np.isfinite(condition) or condition > 1e7:
                break
            delta = np.linalg.solve(normal_matrix + np.eye(3) * 1e-9, rhs)
            transform = matrix(delta[0], delta[1], delta[2]) @ transform
            count = len(selected)
            rmse = float(np.sqrt(np.mean(residual**2)))
            if np.linalg.norm(delta[:2]) < cfg.convergence_translation and abs(delta[2]) < cfg.convergence_yaw:
                break
        x, y, yaw = components(transform)
        ratio = count / max(len(source), 1)
        accepted = (
            count >= cfg.min_correspondences
            and np.hypot(x, y) <= cfg.max_step_translation
            and abs(yaw) <= cfg.max_step_yaw
            and rmse < cfg.huber_delta * 1.5
        )
        return IcpResult(transform, accepted, rmse, ratio, count, iteration + 1, condition)
