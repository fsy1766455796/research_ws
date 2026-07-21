import numpy as np

from orbbec_bev_odometry.multichannel_bev import (
    MultiChannelBEVConfig,
    points_to_multichannel_bev,
)


def test_multichannel_bev_preserves_height_statistics_and_bins():
    config = MultiChannelBEVConfig(resolution_m=0.1)
    points = np.array(
        [[1.0, 0.0, 0.2], [1.0, 0.0, 0.8], [2.0, 1.0, 1.3]],
        dtype=np.float32,
    )
    bev = points_to_multichannel_bev(points, config)
    row = int((1.0 - config.forward_min_m) / config.resolution_m)
    col = int((0.0 - config.left_min_m) / config.resolution_m)

    assert bev.shape[0] == len(config.channel_names)
    assert bev[0, row, col] == 1.0
    assert bev[1, row, col] > 0.0
    assert bev[2, row, col] < bev[3, row, col]
    assert bev[4, row, col] > 0.0
    assert bev[5:, row, col].sum() == 2.0


def test_empty_points_return_zero_bev():
    config = MultiChannelBEVConfig()
    bev = points_to_multichannel_bev(np.empty((0, 3)), config)
    assert bev.shape == (
        len(config.channel_names), config.height, config.width
    )
    assert not np.any(bev)
