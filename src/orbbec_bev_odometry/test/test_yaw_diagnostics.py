import math

import numpy as np
import torch
from torch.nn import functional as F

from orbbec_bev_odometry.learned_bev_model import SE2Correlation, warp_current_to_previous


def test_rotation_and_translation_peak_with_independent_analytic_features():
    bounds = (-1., 6., -3.5, 3.5)
    forward, left = torch.meshgrid(torch.linspace(-.9, 5.9, 35),
                                   torch.linspace(-3.4, 3.4, 35), indexing='ij')
    frequencies = torch.from_numpy(np.random.default_rng(7).uniform(-5, 5, (24, 2))).float()

    def feature(x, y):
        values = torch.sin(frequencies[:, 0, None, None] * x +
                           frequencies[:, 1, None, None] * y +
                           torch.arange(24)[:, None, None])
        return F.normalize(values[None], dim=1)

    previous = feature(forward, left)
    for yaw in (-6., 6.):
        angle = math.radians(yaw)
        current = feature(math.cos(angle) * forward - math.sin(angle) * left + .2,
                          math.sin(angle) * forward + math.cos(angle) * left - .2)
        result = SE2Correlation(bounds)(previous, current)
        index = np.unravel_index(int(result['logits'].argmax()), result['logits'].shape)
        assert abs(math.degrees(float(result['yaw_values'][index[1]])) - yaw) < .01
        assert abs(float(result['forward_values'][index[2]]) - .2) < 1e-6
        assert abs(float(result['left_values'][index[3]]) + .2) < 1e-6


def test_warp_positive_yaw_about_robot_origin():
    # 坐标通道让双线性插值直接返回逆变换采样位置，独立核对旋转中心和符号。
    forward, left = torch.meshgrid(torch.linspace(-.9, 5.9, 35),
                                   torch.linspace(-3.4, 3.4, 35), indexing='ij')
    angle = .1
    current = torch.stack((forward, left))[None]
    pose = torch.tensor([[.2, -.1, angle]])
    output = warp_current_to_previous(current, pose, (-1., 6., -3.5, 3.5))[0]
    x, y = forward - .2, left + .1
    expected = torch.stack((math.cos(angle) * x + math.sin(angle) * y,
                            -math.sin(angle) * x + math.cos(angle) * y))
    torch.testing.assert_close(output[:, 8:25, 8:25], expected[:, 8:25, 8:25], atol=1e-6, rtol=1e-5)
