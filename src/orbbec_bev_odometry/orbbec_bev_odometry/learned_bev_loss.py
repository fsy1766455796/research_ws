"""Training objectives and metrics for learned BEV odometry."""

import math

import torch
from torch.nn import functional as functional


def wrap_angle(angle):
    """Wrap a tensor of angles to the principal interval."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _nearest_index(values, target):
    return torch.argmin(torch.abs(values[None, :] - target[:, None]), dim=1)


def learned_bev_loss(
    output,
    target,
    translation_weight=1.0,
    yaw_weight=2.0,
    correlation_weight=0.25,
    static_weight=0.5,
    step_length_weight=0.5,
    confidence_weight=0.05,
):
    """Pose regression plus supervision of the explicit SE(2) score volume."""
    translation_error = output["pose"][:, :2] - target[:, :2]
    yaw_error = wrap_angle(output["pose"][:, 2] - target[:, 2])
    translation_loss = functional.smooth_l1_loss(
        translation_error, torch.zeros_like(translation_error), beta=0.05
    )
    yaw_loss = functional.smooth_l1_loss(
        yaw_error, torch.zeros_like(yaw_error), beta=math.radians(1.0)
    )

    yaw_index = _nearest_index(output["yaw_values"], target[:, 2])
    forward_index = _nearest_index(output["forward_values"], target[:, 0])
    left_index = _nearest_index(output["left_values"], target[:, 1])
    _, _, forward_count, left_count = output["logits"].shape
    flat_index = (
        yaw_index * forward_count * left_count
        + forward_index * left_count
        + left_index
    )
    correlation_loss = functional.cross_entropy(
        output["logits"].flatten(1), flat_index
    )

    predicted_step = torch.linalg.norm(output["pose"][:, :2], dim=1)
    target_step = torch.linalg.norm(target[:, :2], dim=1)
    # Penalize the exact failure observed with DA3: many small fake steps that
    # inflate travelled distance even when endpoint errors later cancel out.
    step_overrun = functional.relu(predicted_step - target_step - 0.01)
    step_length_loss = (step_overrun * step_overrun).mean()

    stationary = (target_step < 0.015) & (
        torch.abs(target[:, 2]) < math.radians(0.3)
    )
    if torch.any(stationary):
        static_loss = (
            predicted_step[stationary].mean()
            + torch.abs(output["pose"][stationary, 2]).mean()
        )
    else:
        static_loss = output["pose"].sum() * 0.0

    detached_translation_error = torch.linalg.norm(
        output["pose"][:, :2].detach() - target[:, :2], dim=1
    )
    detached_yaw_error = torch.abs(
        wrap_angle(output["pose"][:, 2].detach() - target[:, 2])
    )
    confidence_target = torch.exp(
        -(
            detached_translation_error / 0.10
            + detached_yaw_error / math.radians(5.0)
        )
    ).clamp(0.0, 1.0)
    confidence_loss = functional.binary_cross_entropy(
        output["confidence"], confidence_target
    )
    total = (
        translation_weight * translation_loss
        + yaw_weight * yaw_loss
        + correlation_weight * correlation_loss
        + static_weight * static_loss
        + step_length_weight * step_length_loss
        + confidence_weight * confidence_loss
    )
    return {
        "loss": total,
        "translation_loss": translation_loss.detach(),
        "yaw_loss": yaw_loss.detach(),
        "correlation_loss": correlation_loss.detach(),
        "static_loss": static_loss.detach(),
        "step_length_loss": step_length_loss.detach(),
        "confidence_loss": confidence_loss.detach(),
    }


def pose_error_metrics(prediction, target):
    """Accumulate metric translation and yaw errors for one batch."""
    translation = torch.linalg.norm(prediction[:, :2] - target[:, :2], dim=1)
    yaw = torch.abs(wrap_angle(prediction[:, 2] - target[:, 2]))
    predicted_step = torch.linalg.norm(prediction[:, :2], dim=1)
    target_step = torch.linalg.norm(target[:, :2], dim=1)
    return {
        "translation_sum_m": float(translation.sum().item()),
        "yaw_sum_rad": float(yaw.sum().item()),
        "translation_squared_sum_m2": float((translation * translation).sum().item()),
        "yaw_squared_sum_rad2": float((yaw * yaw).sum().item()),
        "predicted_distance_sum_m": float(predicted_step.sum().item()),
        "target_distance_sum_m": float(target_step.sum().item()),
        "false_step_over_10cm_count": int(torch.count_nonzero(
            (predicted_step > 0.10) & (target_step <= 0.10)
        ).item()),
        "count": int(len(target)),
    }
