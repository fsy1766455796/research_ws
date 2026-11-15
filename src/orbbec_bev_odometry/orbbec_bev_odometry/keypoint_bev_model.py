"""Keypoint matching + weighted 2D Procrustes odometry on multi-channel BEVs.

Route A front end (2026-11-08).  Instead of correlation search plus a
regressed residual, the network predicts per-cell descriptors, keypoint
scores and validity; current-frame keypoints are softly matched into the
previous frame and the SE(2) motion is solved in closed form by weighted
Procrustes.  Everything is differentiable, so the pose loss trains the
descriptors and weights end to end.

Pose convention (same as ``learned_bev_model``): ``[forward, left, yaw]`` of
the current base in the previous base frame, i.e. p_prev = R(yaw) p_cur + t.
BEV rows increase with forward from ``forward_min``; columns increase with
left from ``left_min``.

Design choices for indoor, sparse depth BEVs:
* occupancy prior: keypoints and match targets are restricted to cells that
  contain points (dilated); empty space can neither vote nor attract matches;
* motion-bounded window: a keypoint at range rho may move at most
  t_max + 2 rho sin(yaw_max / 2), so the first matching window grows with range;
* second pass: re-match inside a small radius around the first estimate and
  down-weight keypoints whose matches disagree with it (Cauchy weights);
* additive -inf style masking, so cells outside the window get zero mass.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _block(in_channels, out_channels, stride=1):
    groups = 8 if out_channels % 8 == 0 else 4
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


class BEVUNet(nn.Module):
    """Small U-Net: 140x140 input -> 70x70 (stride 2) dense heads."""

    def __init__(self, input_channels, descriptor_dim=32, width=32):
        super().__init__()
        w = width
        self.enc0 = _block(input_channels, w)            # 140
        self.enc1 = _block(w, 2 * w, stride=2)           # 70
        self.enc2 = _block(2 * w, 3 * w, stride=2)       # 35
        self.enc3 = _block(3 * w, 4 * w, stride=2)       # 18
        self.dec2 = _block(4 * w + 3 * w, 3 * w)         # 35
        self.dec1 = _block(3 * w + 2 * w, 2 * w)         # 70
        self.down0 = nn.Conv2d(w, w, 3, stride=2, padding=1)
        self.fuse = _block(2 * w + w, 2 * w)             # 70
        self.descriptor = nn.Conv2d(2 * w, descriptor_dim, 1)
        self.score = nn.Conv2d(2 * w, 1, 1)
        self.validity = nn.Conv2d(2 * w, 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = self.dec2(torch.cat((F.interpolate(e3, size=e2.shape[-2:], mode="bilinear",
                                                align_corners=False), e2), 1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode="bilinear",
                                                align_corners=False), e1), 1))
        f = self.fuse(torch.cat((d1, self.down0(e0)), 1))
        descriptors = F.normalize(self.descriptor(f), dim=1, eps=1e-6)
        return descriptors, self.score(f), torch.sigmoid(self.validity(f))


def weighted_procrustes_2d(src, tgt, weights, eps=1e-6):
    """Closed-form weighted SE(2) fit tgt ~= R src + t.

    src, tgt: [B, K, 2] metric (forward, left); weights: [B, K] >= 0.
    Returns pose [B, 3] = (t_forward, t_left, yaw) and total weight [B].
    """
    total = weights.sum(dim=1)
    w = weights / (total[:, None] + eps)
    src_mean = (w[..., None] * src).sum(dim=1)
    tgt_mean = (w[..., None] * tgt).sum(dim=1)
    p = src - src_mean[:, None]
    q = tgt - tgt_mean[:, None]
    dot = (w * (p[..., 0] * q[..., 0] + p[..., 1] * q[..., 1])).sum(dim=1)
    cross = (w * (p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0])).sum(dim=1)
    # Tiny positive bias keeps atan2 differentiable for degenerate inputs.
    yaw = torch.atan2(cross, dot + 1e-9)
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    t_forward = tgt_mean[:, 0] - (cosine * src_mean[:, 0] - sine * src_mean[:, 1])
    t_left = tgt_mean[:, 1] - (sine * src_mean[:, 0] + cosine * src_mean[:, 1])
    return torch.stack((t_forward, t_left, yaw), dim=1), total


def apply_pose(pose, points):
    """Map current-frame points [B, K, 2] into the previous frame."""
    cosine = torch.cos(pose[:, 2])[:, None]
    sine = torch.sin(pose[:, 2])[:, None]
    x, y = points[..., 0], points[..., 1]
    return torch.stack((cosine * x - sine * y + pose[:, 0, None],
                        sine * x + cosine * y + pose[:, 1, None]), dim=-1)


class KeypointBEVOdometry(nn.Module):
    """Shared U-Net, soft keypoints, two-pass soft matching, weighted Procrustes."""

    def __init__(
        self,
        input_channels,
        metric_bounds=(-1.0, 6.0, -3.5, 3.5),
        bev_height=140,
        bev_width=140,
        descriptor_dim=32,
        width=32,
        keypoint_patch=2,
        max_translation_m=0.8,
        max_yaw_deg=20.0,
        window_margin_m=0.1,
        refine_radius_m=0.3,
        cauchy_scale_m=0.15,
        initial_temperature=0.05,
        iterations=2,
    ):
        super().__init__()
        self.metric_bounds = tuple(float(v) for v in metric_bounds)
        self.keypoint_patch = int(keypoint_patch)
        self.max_translation_m = float(max_translation_m)
        self.max_yaw = math.radians(max_yaw_deg)
        self.window_margin_m = float(window_margin_m)
        self.refine_radius_m = float(refine_radius_m)
        self.cauchy_scale_m = float(cauchy_scale_m)
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        self.net = BEVUNet(input_channels, descriptor_dim, width)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        # Warm-up switch (official GKP idea): ignore learned validity at first.
        self.uniform_validity = False

        out_h, out_w = (int(bev_height) + 1) // 2, (int(bev_width) + 1) // 2
        if out_h % self.keypoint_patch or out_w % self.keypoint_patch:
            raise ValueError("Output grid must be divisible by keypoint_patch")
        fmin, fmax, lmin, lmax = self.metric_bounds
        self.res_forward = (fmax - fmin) / out_h
        self.res_left = (lmax - lmin) / out_w
        rows = torch.arange(out_h, dtype=torch.float32)
        cols = torch.arange(out_w, dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        cells = torch.stack((fmin + (grid_r + 0.5) * self.res_forward,
                             lmin + (grid_c + 0.5) * self.res_left), dim=-1)
        self.register_buffer("cell_metric", cells.reshape(-1, 2), persistent=False)
        self.register_buffer("cell_rows", grid_r, persistent=False)
        self.register_buffer("cell_cols", grid_c, persistent=False)
        self.out_shape = (out_h, out_w)

    # ------------------------------------------------------------------ utils
    def temperature(self):
        return torch.exp(self.log_temperature).clamp(0.005, 0.5)

    def _normalized(self, metric):
        """Metric (forward, left) -> grid_sample (x=left, y=forward) in [-1, 1]."""
        fmin, fmax, lmin, lmax = self.metric_bounds
        x = 2.0 * (metric[..., 1] - lmin) / (lmax - lmin) - 1.0
        y = 2.0 * (metric[..., 0] - fmin) / (fmax - fmin) - 1.0
        return torch.stack((x, y), dim=-1)

    def _sample(self, image, metric):
        """Bilinear sample [B, C, H, W] at metric points [B, K, 2] -> [B, C, K]."""
        grid = self._normalized(metric)[:, :, None, :]
        return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)[..., 0]

    def _occupancy(self, bev):
        occupied = (bev.abs().sum(dim=1, keepdim=True) > 0).float()
        return F.max_pool2d(occupied, 2, ceil_mode=True)

    def _keypoints(self, score):
        """One soft keypoint per patch; returns metric coords [B, K, 2]."""
        p = self.keypoint_patch
        batch = score.shape[0]
        attention = F.softmax(F.unfold(score, p, stride=p), dim=1)
        rows = F.unfold(self.cell_rows[None, None].expand(batch, -1, -1, -1), p, stride=p)
        cols = F.unfold(self.cell_cols[None, None].expand(batch, -1, -1, -1), p, stride=p)
        row = (attention * rows).sum(dim=1)
        col = (attention * cols).sum(dim=1)
        fmin, _, lmin, _ = self.metric_bounds
        return torch.stack((fmin + (row + 0.5) * self.res_forward,
                            lmin + (col + 0.5) * self.res_left), dim=-1)

    # ---------------------------------------------------------------- forward
    def _match(self, logits, allowed, prev_desc, prev_val, kp_desc):
        """Soft-assign within ``allowed``; return targets, base weights, valid rows."""
        has_candidate = allowed.any(dim=-1)
        masked = logits.masked_fill(~allowed, -1e4)
        probability = torch.softmax(masked, dim=-1)
        target = probability @ self.cell_metric
        target_desc = F.normalize(self._sample(prev_desc, target), dim=1, eps=1e-6)
        similarity = (kp_desc * target_desc).sum(dim=1)
        weight = 0.5 * (1.0 + similarity)
        if not self.uniform_validity:
            weight = weight * self._sample(prev_val, target)[:, 0]
        return target, weight * has_candidate.float(), has_candidate

    def forward(self, previous_bev, current_bev):
        if previous_bev.shape != current_bev.shape:
            raise ValueError("previous_bev and current_bev must have equal shape")
        batch = previous_bev.shape[0]
        both = torch.cat((previous_bev, current_bev), dim=0).float()
        descriptors, scores, validity = self.net(both)
        occupancy = self._occupancy(both)
        prev_desc, cur_desc = descriptors[:batch], descriptors[batch:]
        prev_val, cur_val = validity[:batch], validity[batch:]
        prev_occ, cur_occ = occupancy[:batch], occupancy[batch:]
        cur_score = scores[batch:]

        with torch.autocast(device_type=both.device.type, enabled=False):
            prev_desc, cur_desc = prev_desc.float(), cur_desc.float()
            prev_val, cur_val = prev_val.float(), cur_val.float()
            keypoints = self._keypoints(cur_score.float())                 # [B,K,2]
            kp_desc = F.normalize(self._sample(cur_desc, keypoints), dim=1, eps=1e-6)
            kp_occ = self._sample(cur_occ, keypoints)[:, 0].clamp(0, 1)   # soft prior
            kp_weight = kp_occ if self.uniform_validity else kp_occ * self._sample(cur_val, keypoints)[:, 0]

            target_ok = F.max_pool2d(prev_occ, 3, stride=1, padding=1).flatten(1) > 0  # [B,HW]
            dense = prev_desc.flatten(2)                                     # [B,D,HW]
            logits = torch.einsum("bdk,bdn->bkn", kp_desc, dense) / self.temperature()
            cells = self.cell_metric[None, None]                            # [1,1,HW,2]

            rho = keypoints.norm(dim=-1, keepdim=True)
            radius = (self.max_translation_m + self.window_margin_m
                      + 2.0 * rho * math.sin(self.max_yaw / 2.0))
            distance = (cells - keypoints[:, :, None]).norm(dim=-1)
            allowed = (distance <= radius) & target_ok[:, None]
            target, match_weight, _ = self._match(logits, allowed, prev_desc, prev_val, kp_desc)
            weights = kp_weight * match_weight
            pose, total = weighted_procrustes_2d(keypoints, target, weights)
            passes = [dict(pose=pose, target=target, weights=weights)]

            for _ in range(self.iterations - 1):
                predicted = apply_pose(pose.detach(), keypoints)
                distance = (cells - predicted[:, :, None]).norm(dim=-1)
                allowed = (distance <= self.refine_radius_m) & target_ok[:, None]
                target, match_weight, _ = self._match(logits, allowed, prev_desc, prev_val, kp_desc)
                residual = (target - predicted).norm(dim=-1).detach()
                cauchy = 1.0 / (1.0 + (residual / self.cauchy_scale_m) ** 2)
                weights = kp_weight * match_weight * cauchy
                refined, total = weighted_procrustes_2d(keypoints, target, weights)
                usable = (total > 1e-3)[:, None]
                pose = torch.where(usable, refined, pose)
                passes.append(dict(pose=pose, target=target, weights=weights))

        confidence = 1.0 - torch.exp(-total / 5.0)
        return {
            "pose": pose,
            "confidence": confidence.clamp(0.0, 1.0),
            "passes": passes,
            "keypoints": keypoints,
            "keypoint_occupancy": kp_occ,
            "previous_occupancy": prev_occ,
            "temperature": self.temperature().detach(),
            "metric_bounds": self.metric_bounds,
        }
