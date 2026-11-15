import math

import torch

from orbbec_bev_odometry.keypoint_bev_loss import keypoint_bev_loss
from orbbec_bev_odometry.keypoint_bev_model import (
    KeypointBEVOdometry, apply_pose, weighted_procrustes_2d)
from orbbec_bev_odometry.learned_bev_dataset import invert_planar_target
from orbbec_bev_odometry.learned_bev_model import warp_current_to_previous


def _random_pose(batch, generator):
    pose = torch.empty(batch, 3)
    pose[:, :2] = (torch.rand(batch, 2, generator=generator) - 0.5) * 1.2
    pose[:, 2] = (torch.rand(batch, generator=generator) - 0.5) * math.radians(36)
    return pose


def test_procrustes_recovers_pose_and_ignores_zero_weight_outliers():
    g = torch.Generator().manual_seed(0)
    pose = _random_pose(8, g)
    src = torch.rand(8, 50, 2, generator=g) * 6 - torch.tensor([1.0, 3.0])
    tgt = apply_pose(pose, src)
    weights = torch.rand(8, 50, generator=g) + 0.1
    tgt[:, :10] += 3.0
    weights[:, :10] = 0.0
    estimate, _ = weighted_procrustes_2d(src, tgt, weights)
    assert torch.allclose(estimate, pose, atol=1e-4)


def test_apply_pose_matches_inverse_target():
    g = torch.Generator().manual_seed(1)
    pose = _random_pose(4, g)
    points = torch.rand(4, 20, 2, generator=g) * 4
    inverse = torch.stack([torch.from_numpy(invert_planar_target(p.numpy())) for p in pose])
    back = apply_pose(inverse, apply_pose(pose, points))
    assert torch.allclose(back, points, atol=1e-5)


def test_pose_convention_matches_existing_warp():
    """Occupied current cells mapped by the pose must land on occupied previous cells."""
    g = torch.Generator().manual_seed(2)
    previous = torch.zeros(1, 1, 140, 140)
    rows = torch.randint(30, 120, (400,), generator=g)
    cols = torch.randint(20, 120, (400,), generator=g)
    previous[0, 0, rows, cols] = 1.0
    previous = torch.nn.functional.max_pool2d(previous, 5, 1, 2)
    pose = torch.tensor([[0.3, -0.2, math.radians(12.0)]])
    inverse = torch.from_numpy(invert_planar_target(pose[0].numpy()))[None]
    current = warp_current_to_previous(previous, inverse, (-1.0, 6.0, -3.5, 3.5))
    occupied = torch.nonzero(current[0, 0] > 0.99)
    metric = torch.stack((-1.0 + (occupied[:, 0].float() + 0.5) * 0.05,
                          -3.5 + (occupied[:, 1].float() + 0.5) * 0.05), -1)[None]
    mapped = apply_pose(pose, metric)
    grid = torch.stack((2 * (mapped[..., 1] + 3.5) / 7 - 1, 2 * (mapped[..., 0] + 1) / 7 - 1), -1)[:, :, None]
    hit = torch.nn.functional.grid_sample(previous, grid, align_corners=False)[0, 0, :, 0]
    assert occupied.shape[0] > 100
    assert (hit > 0.5).float().mean() > 0.95


def test_forward_backward_finite_including_empty_bev():
    torch.manual_seed(0)
    model = KeypointBEVOdometry(input_channels=9)
    previous = (torch.rand(2, 9, 140, 140) > 0.95).float()
    current = previous.clone()
    previous[1].zero_()
    current[1].zero_()
    target = torch.tensor([[0.1, 0.0, 0.05], [0.0, 0.0, 0.0]])
    output = model(previous, current)
    loss = keypoint_bev_loss(output, target)["loss"]
    loss.backward()
    assert torch.isfinite(output["pose"]).all()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
