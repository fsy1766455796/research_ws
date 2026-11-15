"""Height-aware BEV registration network with an explicit SE(2) matcher."""

import math

import torch
from torch import nn
from torch.nn import functional as functional


class ResidualBlock(nn.Module):
    """Preserve spatial resolution while refining BEV features."""

    def __init__(self, channels):
        """Create a two-convolution residual block."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs):
        """Apply the residual block."""
        return self.activation(inputs + self.layers(inputs))


class SharedBEVEncoder(nn.Module):
    """A small shared encoder returning stride-2 and stride-4 features."""

    def __init__(self, input_channels, fine_channels=32, coarse_channels=24):
        """Create shared fine and coarse feature stages."""
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, fine_channels, 5, padding=2, bias=False),
            nn.GroupNorm(4, fine_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(fine_channels),
        )
        self.fine = nn.Sequential(
            nn.Conv2d(
                fine_channels, fine_channels, 3, stride=2, padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, fine_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(fine_channels),
        )
        self.coarse = nn.Sequential(
            nn.Conv2d(
                fine_channels, coarse_channels, 3, stride=2, padding=1,
                bias=False,
            ),
            nn.GroupNorm(4, coarse_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(coarse_channels),
        )

    def forward(self, inputs):
        """Return normalized stride-2 and stride-4 features."""
        fine = self.fine(self.stem(inputs))
        coarse = self.coarse(fine)
        fine = functional.normalize(fine, dim=1, eps=1e-6)
        coarse = functional.normalize(coarse, dim=1, eps=1e-6)
        return fine, coarse


def warp_current_to_previous(current, pose, metric_bounds):
    """Warp current-frame features into the previous frame.

    ``pose`` is ``[forward, left, yaw]`` and maps current robot coordinates to
    previous robot coordinates.  Rotation is performed around the robot origin,
    which need not coincide with the centre of the BEV crop.
    """
    if current.ndim != 4 or pose.ndim != 2 or pose.shape[1] != 3:
        raise ValueError("Expected current [B,C,H,W] and pose [B,3]")
    forward_min, forward_max, left_min, left_max = metric_bounds
    batch, _, height, width = current.shape
    forward_axis = torch.linspace(
        forward_min + (forward_max - forward_min) / (2.0 * height),
        forward_max - (forward_max - forward_min) / (2.0 * height),
        height,
        dtype=current.dtype,
        device=current.device,
    )
    left_axis = torch.linspace(
        left_min + (left_max - left_min) / (2.0 * width),
        left_max - (left_max - left_min) / (2.0 * width),
        width,
        dtype=current.dtype,
        device=current.device,
    )
    forward, left = torch.meshgrid(
        forward_axis, left_axis, indexing="ij"
    )
    forward = forward.unsqueeze(0).expand(batch, -1, -1)
    left = left.unsqueeze(0).expand(batch, -1, -1)

    translated_forward = forward - pose[:, 0, None, None]
    translated_left = left - pose[:, 1, None, None]
    cosine = torch.cos(pose[:, 2, None, None])
    sine = torch.sin(pose[:, 2, None, None])
    # Inverse of p_previous = R(yaw) p_current + translation.
    sample_forward = cosine * translated_forward + sine * translated_left
    sample_left = -sine * translated_forward + cosine * translated_left
    normalized_x = (
        2.0 * (sample_left - left_min) / (left_max - left_min) - 1.0
    )
    normalized_y = (
        2.0 * (sample_forward - forward_min)
        / (forward_max - forward_min) - 1.0
    )
    grid = torch.stack((normalized_x, normalized_y), dim=-1)
    return functional.grid_sample(
        current,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class SE2Correlation(nn.Module):
    """Discrete yaw search plus linear FFT correlation for metric translation."""

    def __init__(
        self,
        metric_bounds,
        max_translation_m=0.8,
        max_yaw_deg=20.0,
        yaw_bins=41,
        correlation_mode="legacy",
    ):
        """Configure metric translation and yaw search bounds."""
        super().__init__()
        if yaw_bins < 3 or yaw_bins % 2 == 0:
            raise ValueError("yaw_bins must be an odd integer of at least 3")
        self.metric_bounds = tuple(float(value) for value in metric_bounds)
        self.max_translation_m = float(max_translation_m)
        if correlation_mode not in ("legacy", "cosine"):
            raise ValueError("correlation_mode must be legacy or cosine")
        self.correlation_mode = correlation_mode
        yaw_values = torch.linspace(
            -math.radians(max_yaw_deg), math.radians(max_yaw_deg), yaw_bins
        )
        self.register_buffer("yaw_values", yaw_values)
        initial = math.log(10.0) if correlation_mode == "legacy" else math.log(9.0 / 191.0)
        self.logit_scale = nn.Parameter(torch.tensor(initial))

    @staticmethod
    def _linear_correlation(previous, current):
        """Return correlations whose central cell represents zero shift."""
        _, _, height, width = previous.shape
        padded_shape = (2 * height - 1, 2 * width - 1)
        previous_fft = torch.fft.rfft2(previous, s=padded_shape)
        current_fft = torch.fft.rfft2(current, s=padded_shape)
        cross_power = (previous_fft * current_fft.conj()).sum(dim=1)
        correlation = torch.fft.irfft2(cross_power, s=padded_shape)
        return torch.roll(
            correlation, shifts=(height - 1, width - 1), dims=(-2, -1)
        )

    def forward(self, previous, current):
        """Score all configured yaw and local translation hypotheses."""
        batch, channels, height, width = previous.shape
        forward_min, forward_max, left_min, left_max = self.metric_bounds
        forward_resolution = (forward_max - forward_min) / height
        left_resolution = (left_max - left_min) / width
        forward_shift_count = int(
            math.floor(self.max_translation_m / forward_resolution)
        )
        left_shift_count = int(
            math.floor(self.max_translation_m / left_resolution)
        )

        yaw_count = len(self.yaw_values)
        expanded_current = current[:, None].expand(
            -1, yaw_count, -1, -1, -1
        ).reshape(batch * yaw_count, channels, height, width)
        yaw_pose = torch.zeros(
            (batch, yaw_count, 3), dtype=current.dtype, device=current.device
        )
        yaw_pose[:, :, 2] = self.yaw_values.to(dtype=current.dtype)
        rotated = warp_current_to_previous(
            expanded_current,
            yaw_pose.reshape(batch * yaw_count, 3),
            self.metric_bounds,
        ).reshape(batch, yaw_count, channels, height, width)

        previous_expanded = previous[:, None].expand(
            -1, yaw_count, -1, -1, -1
        ).reshape(batch * yaw_count, channels, height, width)
        correlation = self._linear_correlation(
            previous_expanded,
            rotated.reshape(batch * yaw_count, channels, height, width),
        ).reshape(batch, yaw_count, 2 * height - 1, 2 * width - 1)
        centre_row, centre_col = height - 1, width - 1
        correlation = correlation[
            :,
            :,
            centre_row - forward_shift_count:
            centre_row + forward_shift_count + 1,
            centre_col - left_shift_count:centre_col + left_shift_count + 1,
        ]
        if self.correlation_mode == "legacy":
            scale = torch.exp(self.logit_scale).clamp(max=100.0)
            logits = correlation * scale / max(channels * height * width, 1)
        else:
            # Channel-normalized dot products are already cosine similarities.
            # A smooth bounded temperature avoids the legacy hard-clamp dead zone.
            scale = 1.0 + 200.0 * torch.sigmoid(self.logit_scale)
            logits = correlation * scale / max(height * width, 1)
        probability = torch.softmax(logits.flatten(1), dim=1).reshape_as(logits)

        forward_values = torch.arange(
            -forward_shift_count,
            forward_shift_count + 1,
            device=current.device,
            dtype=current.dtype,
        ) * forward_resolution
        left_values = torch.arange(
            -left_shift_count,
            left_shift_count + 1,
            device=current.device,
            dtype=current.dtype,
        ) * left_resolution
        yaw_values = self.yaw_values.to(dtype=current.dtype)
        yaw_grid, forward_grid, left_grid = torch.meshgrid(
            yaw_values, forward_values, left_values, indexing="ij"
        )
        coarse_pose = torch.stack(
            (
                (probability * forward_grid).sum(dim=(1, 2, 3)),
                (probability * left_grid).sum(dim=(1, 2, 3)),
                (probability * yaw_grid).sum(dim=(1, 2, 3)),
            ),
            dim=1,
        )
        entropy = -(
            probability.flatten(1)
            * torch.log(probability.flatten(1).clamp_min(1e-9))
        ).sum(dim=1)
        confidence = 1.0 - entropy / math.log(probability[0].numel())
        return {
            "pose": coarse_pose,
            "logits": logits,
            "probability": probability,
            "confidence": confidence.clamp(0.0, 1.0),
            "forward_values": forward_values,
            "left_values": left_values,
            "yaw_values": yaw_values,
        }


class FinePoseRefiner(nn.Module):
    """Predict a bounded sub-bin correction after coarse matching."""

    def __init__(self, feature_channels, translation_limit, yaw_limit):
        """Create the feature fusion and bounded regression head."""
        super().__init__()
        fused_channels = feature_channels * 4
        self.feature_network = nn.Sequential(
            nn.Conv2d(fused_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.pose_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 3),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )
        self.register_buffer(
            "limits",
            torch.tensor(
                [translation_limit, translation_limit, yaw_limit],
                dtype=torch.float32,
            ),
        )

    def forward(self, previous, aligned_current):
        """Regress a small SE(2) residual from aligned fine features."""
        fused = torch.cat(
            (
                previous,
                aligned_current,
                previous - aligned_current,
                previous * aligned_current,
            ),
            dim=1,
        )
        pair_feature = self.feature_network(fused)
        residual = torch.tanh(self.pose_head(pair_feature)) * self.limits
        confidence = torch.sigmoid(self.confidence_head(pair_feature)).squeeze(1)
        return residual, confidence, pair_feature


class LearnedBEVOdometry(nn.Module):
    """Shared BEV encoder, explicit SE(2) search and sub-bin refinement."""

    def __init__(
        self,
        input_channels,
        metric_bounds=(-1.0, 6.0, -3.5, 3.5),
        max_translation_m=0.8,
        max_yaw_deg=20.0,
        yaw_bins=41,
        bev_height=140,
        bev_width=140,
        yaw_residual_deg=None,
        correlation_mode="legacy",
    ):
        """Create the encoder, explicit matcher and fine refiner."""
        super().__init__()
        self.input_channels = int(input_channels)
        self.metric_bounds = tuple(float(value) for value in metric_bounds)
        self.encoder = SharedBEVEncoder(self.input_channels)
        self.matcher = SE2Correlation(
            self.metric_bounds,
            max_translation_m=max_translation_m,
            max_yaw_deg=max_yaw_deg,
            yaw_bins=yaw_bins,
            correlation_mode=correlation_mode,
        )
        coarse_height = ((int(bev_height) + 1) // 2 + 1) // 2
        coarse_width = ((int(bev_width) + 1) // 2 + 1) // 2
        coarse_cell_m = max(
            (self.metric_bounds[1] - self.metric_bounds[0]) / coarse_height,
            (self.metric_bounds[3] - self.metric_bounds[2]) / coarse_width,
        )
        yaw_step = 2.0 * math.radians(max_yaw_deg) / (yaw_bins - 1)
        if yaw_residual_deg is not None:
            if not math.isfinite(yaw_residual_deg) or yaw_residual_deg <= 0:
                raise ValueError("yaw_residual_deg must be finite and positive")
            yaw_step = math.radians(yaw_residual_deg)
        self.refiner = FinePoseRefiner(
            feature_channels=32,
            translation_limit=coarse_cell_m,
            yaw_limit=yaw_step,
        )

    def forward(self, previous_bev, current_bev):
        """Estimate current base pose in the previous base frame."""
        if previous_bev.shape != current_bev.shape:
            raise ValueError("previous_bev and current_bev must have equal shape")
        previous_fine, previous_coarse = self.encoder(previous_bev)
        current_fine, current_coarse = self.encoder(current_bev)
        match = self.matcher(previous_coarse, current_coarse)
        aligned_current = warp_current_to_previous(
            current_fine, match["pose"], self.metric_bounds
        )
        residual, learned_confidence, pair_feature = self.refiner(
            previous_fine, aligned_current
        )
        result = dict(match)
        result["correlation_confidence"] = match["confidence"]
        result["coarse_pose"] = match["pose"]
        result["residual"] = residual
        result["pose"] = match["pose"] + residual
        result["confidence"] = learned_confidence
        result["pair_feature"] = pair_feature
        return result
