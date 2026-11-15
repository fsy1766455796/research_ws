"""Training objective for ``KeypointBEVOdometry``."""

import math

import torch
from torch.nn import functional as F

from .keypoint_bev_model import apply_pose
from .learned_bev_loss import wrap_angle


def _pose_loss(pose, target, yaw_weight):
    translation = F.smooth_l1_loss(pose[:, :2], target[:, :2], beta=0.05)
    yaw_error = wrap_angle(pose[:, 2] - target[:, 2])
    yaw = F.smooth_l1_loss(yaw_error, torch.zeros_like(yaw_error), beta=math.radians(1.0))
    return translation, yaw


def keypoint_bev_loss(output, target, yaw_weight=4.0, first_pass_weight=0.5,
                      correspondence_weight=0.2, correspondence_clip_m=0.5):
    """Pose loss on every matching pass plus clipped keypoint correspondence loss.

    The correspondence term uses the ground-truth motion to place each
    current keypoint in the previous frame and only supervises keypoints that
    lie on occupied current cells and land on (dilated) occupied previous cells,
    so free space and newly visible structure are not forced to match.
    """
    target = target.float()
    passes = output["passes"]
    with torch.no_grad():
        expected = apply_pose(target, output["keypoints"])
        previous_occupied = F.max_pool2d(output["previous_occupancy"].float(), 3, 1, 1)
        fmin, fmax, lmin, lmax = output["metric_bounds"]
        grid = torch.stack((2 * (expected[..., 1] - lmin) / (lmax - lmin) - 1,
                            2 * (expected[..., 0] - fmin) / (fmax - fmin) - 1), -1)[:, :, None]
        lands = F.grid_sample(previous_occupied, grid, align_corners=False)[:, 0, :, 0] > 0.5
        visible = (output["keypoint_occupancy"] > 0.5) & lands
        visible_count = visible.float().sum().clamp_min(1.0)

    total = 0.0
    parts = {}
    for index, item in enumerate(passes):
        weight = 1.0 if index == len(passes) - 1 else first_pass_weight
        translation, yaw = _pose_loss(item["pose"], target, yaw_weight)
        error = (item["target"] - expected).norm(dim=-1).clamp(max=correspondence_clip_m)
        correspondence = (error * visible.float()).sum() / visible_count
        total = total + weight * (translation + yaw_weight * yaw
                                  + correspondence_weight * correspondence)
        parts[f"pass{index}_translation"] = translation.detach()
        parts[f"pass{index}_yaw"] = yaw.detach()
        parts[f"pass{index}_correspondence"] = correspondence.detach()
    parts["visible_keypoints"] = visible.float().sum(dim=1).mean().detach()
    parts["loss"] = total
    return parts
