"""Metric multi-channel BEV construction for learned planar registration.

The learned odometry path uses one coordinate convention throughout:

* ``x``: robot forward, in metres;
* ``y``: robot left, in metres;
* ``z``: robot up, in metres.

The returned tensor is channel-first ``[C, H, W]``. Rows increase in the
forward direction and columns increase to the left.  Keeping the metric
geometry explicit makes pose labels and correlation-bin conversions auditable.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


OPENLORIS_COLOR_INTRINSICS = (611.4509887695312, 611.4857177734375,
                              433.2039794921875, 249.4730224609375)
OPENLORIS_COLOR_SIZE = (848, 480)

# T_base_link_d400_color from the calibration released with OpenLORIS-Scene.
# It maps D400 optical coordinates (right, down, forward) into base_link.
OPENLORIS_T_BASE_COLOR = np.array(
    [
        [0.009979281625, 0.006534810370, 0.999928852565, 0.22648368],
        [-0.999820146584, 0.016192923277, 0.009872371528, -0.05114194],
        [-0.016127257116, -0.999847531121, 0.006695228804, 0.91600000],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class MultiChannelBEVConfig:
    """Metric bounds and feature layout for one local BEV."""

    forward_min_m: float = -1.0
    forward_max_m: float = 6.0
    left_min_m: float = -3.5
    left_max_m: float = 3.5
    height_min_m: float = -0.2
    height_max_m: float = 1.8
    resolution_m: float = 0.05
    density_clip: int = 32
    height_bins_m: Tuple[float, ...] = (-0.2, 0.25, 0.75, 1.25, 1.8)

    @property
    def height(self):
        """Return the BEV row count."""
        return int(np.ceil(
            (self.forward_max_m - self.forward_min_m) / self.resolution_m
        ))

    @property
    def width(self):
        """Return the BEV column count."""
        return int(np.ceil(
            (self.left_max_m - self.left_min_m) / self.resolution_m
        ))

    @property
    def origin_row(self):
        """Continuous BEV row coordinate of the robot origin."""
        return -self.forward_min_m / self.resolution_m - 0.5

    @property
    def origin_col(self):
        """Continuous BEV column coordinate of the robot origin."""
        return -self.left_min_m / self.resolution_m - 0.5

    @property
    def channel_names(self):
        """Return channel names in tensor order."""
        base = ("occupancy", "density", "height_mean", "height_max",
                "height_variance")
        bins = tuple(
            f"height_bin_{i}" for i in range(len(self.height_bins_m) - 1)
        )
        return base + bins


def scale_intrinsics(intrinsics, source_size, target_size):
    """Scale ``fx, fy, cx, cy`` when a depth map was resized."""
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    source_width, source_height = (float(value) for value in source_size)
    target_width, target_height = (float(value) for value in target_size)
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def depth_to_robot_points(
    depth_m: np.ndarray,
    intrinsics: Sequence[float] = OPENLORIS_COLOR_INTRINSICS,
    t_base_camera: Optional[np.ndarray] = OPENLORIS_T_BASE_COLOR,
    pixel_stride: int = 2,
    min_depth_m: float = 0.2,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Back-project a metric depth image into ``[forward, left, up]`` points.

    ``intrinsics`` is ``(fx, fy, cx, cy)`` and must match the image to which
    the depth is aligned.  Passing ``t_base_camera=None`` applies the standard
    optical-to-robot axis conversion without a camera mounting transform.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional array")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be at least 1")

    fx, fy, cx, cy = (float(v) for v in intrinsics)
    rows = np.arange(0, depth.shape[0], pixel_stride, dtype=np.float32)
    cols = np.arange(0, depth.shape[1], pixel_stride, dtype=np.float32)
    u, v = np.meshgrid(cols, rows)
    z = depth[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    z = z[valid]
    x_right = (u[valid] - cx) * z / fx
    y_down = (v[valid] - cy) * z / fy
    optical = np.column_stack((x_right, y_down, z)).astype(np.float64)

    if t_base_camera is None:
        robot = np.column_stack((optical[:, 2], -optical[:, 0],
                                 -optical[:, 1]))
    else:
        transform = np.asarray(t_base_camera, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("t_base_camera must have shape (4, 4)")
        robot = optical @ transform[:3, :3].T + transform[:3, 3]
    return robot.astype(np.float32)


def points_to_multichannel_bev(
    points_forward_left_up: np.ndarray,
    config: Optional[MultiChannelBEVConfig] = None,
) -> np.ndarray:
    """Rasterize robot-frame points into occupancy, density and height maps."""
    cfg = config or MultiChannelBEVConfig()
    points = np.asarray(points_forward_left_up, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    output = np.zeros(
        (len(cfg.channel_names), cfg.height, cfg.width), dtype=np.float32
    )
    if len(points) == 0:
        return output

    finite = np.isfinite(points).all(axis=1)
    forward, left, up = points.T
    valid = finite
    valid &= forward >= cfg.forward_min_m
    valid &= forward < cfg.forward_max_m
    valid &= left >= cfg.left_min_m
    valid &= left < cfg.left_max_m
    valid &= up >= cfg.height_min_m
    valid &= up < cfg.height_max_m
    if not np.any(valid):
        return output

    forward = forward[valid]
    left = left[valid]
    up = up[valid]
    rows = np.floor(
        (forward - cfg.forward_min_m) / cfg.resolution_m
    ).astype(np.int32)
    cols = np.floor((left - cfg.left_min_m) / cfg.resolution_m).astype(
        np.int32
    )

    counts = np.zeros((cfg.height, cfg.width), dtype=np.float32)
    height_sum = np.zeros_like(counts)
    height_square_sum = np.zeros_like(counts)
    height_max = np.full_like(counts, -np.inf)
    np.add.at(counts, (rows, cols), 1.0)
    np.add.at(height_sum, (rows, cols), up)
    np.add.at(height_square_sum, (rows, cols), up * up)
    np.maximum.at(height_max, (rows, cols), up)

    occupied = counts > 0
    height_range = max(cfg.height_max_m - cfg.height_min_m, 1e-6)
    mean = np.zeros_like(counts)
    variance = np.zeros_like(counts)
    mean[occupied] = height_sum[occupied] / counts[occupied]
    variance[occupied] = np.maximum(
        height_square_sum[occupied] / counts[occupied]
        - mean[occupied] * mean[occupied],
        0.0,
    )

    output[0] = occupied.astype(np.float32)
    output[1] = np.log1p(np.minimum(counts, cfg.density_clip)) / np.log1p(
        cfg.density_clip
    )
    output[2, occupied] = np.clip(
        (mean[occupied] - cfg.height_min_m) / height_range, 0.0, 1.0
    )
    output[3, occupied] = np.clip(
        (height_max[occupied] - cfg.height_min_m) / height_range, 0.0, 1.0
    )
    output[4, occupied] = np.clip(
        variance[occupied] / (height_range * height_range), 0.0, 1.0
    )

    for index, (low, high) in enumerate(
        zip(cfg.height_bins_m[:-1], cfg.height_bins_m[1:]), start=5
    ):
        in_bin = (up >= low) & (up < high)
        if np.any(in_bin):
            np.maximum.at(output[index], (rows[in_bin], cols[in_bin]), 1.0)
    return output


def depth_to_multichannel_bev(
    depth_m: np.ndarray,
    config: Optional[MultiChannelBEVConfig] = None,
    intrinsics: Sequence[float] = OPENLORIS_COLOR_INTRINSICS,
    t_base_camera: Optional[np.ndarray] = OPENLORIS_T_BASE_COLOR,
    pixel_stride: int = 2,
) -> np.ndarray:
    """Compose depth back-projection and BEV rasterization."""
    points = depth_to_robot_points(
        depth_m,
        intrinsics=intrinsics,
        t_base_camera=t_base_camera,
        pixel_stride=pixel_stride,
    )
    return points_to_multichannel_bev(points, config=config)
