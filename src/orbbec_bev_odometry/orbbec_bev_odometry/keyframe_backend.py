"""Keyframe pose-graph backend (v2) for BEV odometry; no ground truth is used.

Why v2: the 8-frame sliding window of ``local_map_backend`` remembers only
about 1.6 s and caps corrections at 0.12 m / 4 deg, so it can smooth jitter
but cannot remove accumulated yaw drift.  Here:

1. Keyframes are created after 0.25 m or 8 deg of estimated motion (or every
   ``max_gap`` frames).  Non-keyframes are attached to their preceding keyframe
   by the frontend increments.
2. Each new keyframe is registered by ICP against each of the previous
   ``neighbors`` keyframes individually (multi-edge constraints spanning several
   metres), not only against a short-lived merged map.
3. Loop candidates: older keyframes outside that neighbourhood whose estimated
   pose is within ``loop_radius_m`` and ``loop_yaw_deg``; accepted only with a
   stricter gate.
4. After each keyframe the whole keyframe graph is re-optimized (robust
   least squares, first keyframe fixed).

Odometry edges between keyframes use sigma growing with sqrt(number of frames).
Scale bias of the input (e.g. DA3 depth that is too far) cannot be removed by
self-consistent scan matching; that must be fixed in the frontend.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .local_map_backend import (BackendConfig, compose, load_points, metrics, pair_key,
                                relative, select_chains, write_csv)
from .planar_icp import IcpConfig, PlanarPointToPlaneICP
from .se2 import components, matrix, transform_points, wrap_angle


@dataclass
class KeyframeConfig:
    keyframe_distance_m: float = 0.25
    keyframe_angle_deg: float = 8.0
    max_gap: int = 10
    neighbors: int = 8
    loop_radius_m: float = 2.0
    loop_yaw_deg: float = 45.0
    voxel_m: float = 0.06
    max_points: int = 1500
    min_points: int = 40
    correspondence_m: float = 0.15
    min_overlap: float = 0.35
    max_rmse_m: float = 0.07
    max_condition: float = 1e5
    max_correction_m: float = 0.30
    max_correction_deg: float = 12.0
    icp_iterations: int = 60
    loop_min_overlap: float = 0.55
    loop_max_rmse_m: float = 0.05
    odom_sigma_m: float = 0.02
    odom_sigma_deg: float = 0.6
    icp_sigma_m: float = 0.03
    icp_sigma_deg: float = 0.6
    robust_scale: float = 2.0
    learned_max_m: float = 0.30
    learned_max_deg: float = 15.0
    learned_sigma_m: float = 0.02
    learned_sigma_deg: float = 0.5
    learned_consistency_m: float = 0.03
    learned_consistency_deg: float = 0.5


def edge_information(source, target, measured, cfg):
    """Square-root information of an ICP edge in relative-pose error space.

    Built from the point-to-line Jacobian at the solution, so a scan that sees
    only one wall constrains yaw and the across-wall offset but contributes
    (almost) nothing along the wall, instead of being discarded as degenerate.
    Normalized by the number of correspondences (points are not independent):
    for isotropic structure this gives ~icp_sigma_m in translation.
    """
    moved = transform_points(matrix(*measured), source)
    tree = cKDTree(target)
    distances, indices = tree.query(moved)
    keep = distances < cfg.correspondence_m
    if keep.sum() < cfg.min_points:
        return None
    _, neighbors = tree.query(target[indices[keep]], k=min(10, len(target)))
    hood = target[neighbors] - target[neighbors].mean(axis=1, keepdims=True)
    _, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', hood, hood))
    normals = vectors[:, :, 0]
    p = moved[keep]
    jacobian = np.column_stack((normals[:, 0], normals[:, 1],
                                normals[:, 1] * p[:, 0] - normals[:, 0] * p[:, 1]))
    info_delta = 2.0 * (jacobian.T @ jacobian) / len(p) / cfg.icp_sigma_m ** 2
    # Left perturbation delta -> pose-component error e = A delta.
    tx, ty = measured[0], measured[1]
    a = np.array([[1.0, 0.0, -ty], [0.0, 1.0, tx], [0.0, 0.0, 1.0]])
    a_inv = np.linalg.inv(a)
    info = a_inv.T @ info_delta @ a_inv
    info = 0.5 * (info + info.T) + np.eye(3) * 1e-6
    yaw_floor = 1.0 / math.radians(cfg.icp_sigma_deg) ** 2
    info[2, 2] = min(info[2, 2], yaw_floor * 4.0)   # do not trust yaw beyond sigma/2
    return np.linalg.cholesky(info).T


def register_pair(source, target, initial, cfg, loop=False):
    """ICP source(local) -> target(local); returns measured relative pose or None."""
    info = dict(accepted=False)
    if min(len(source), len(target)) < cfg.min_points:
        info['reason'] = 'too_few_points'
        return None, info
    # Point-to-point ICP converges slowly in rotation; with the default 18
    # iterations it stopped near the initial guess and under-corrected yaw.
    solver = PlanarPointToPlaneICP(IcpConfig(
        min_correspondences=cfg.min_points, max_correspondence=0.25,
        max_iterations=cfg.icp_iterations, convergence_translation=1e-5,
        convergence_yaw=1e-6, max_step_translation=np.inf, max_step_yaw=np.inf))
    result = solver.register_point_to_point(source, target, initial=matrix(*initial))
    candidate = np.asarray(components(result.transform))
    distances, indices = cKDTree(target).query(transform_points(result.transform, source))
    mask = distances < cfg.correspondence_m
    overlap = float(mask.mean())
    rmse = float(np.sqrt(np.mean(distances[mask] ** 2))) if mask.any() else math.inf
    condition = math.inf
    if mask.sum() >= cfg.min_points:
        eig = np.linalg.eigvalsh(np.cov(target[indices[mask]].T))
        if eig[0] > 1e-10:
            condition = float(eig[-1] / eig[0])
    delta = relative(initial, candidate)
    correction_m, correction_deg = float(np.hypot(*delta[:2])), abs(math.degrees(delta[2]))
    min_overlap = cfg.loop_min_overlap if loop else cfg.min_overlap
    max_rmse = cfg.loop_max_rmse_m if loop else cfg.max_rmse_m
    # Loops still require well-conditioned structure; neighbour edges rely on
    # the information matrix to down-weight degenerate directions.
    accepted = (np.isfinite(rmse) and overlap >= min_overlap and rmse <= max_rmse
                and (not loop or condition <= cfg.max_condition)
                and correction_m <= cfg.max_correction_m
                and correction_deg <= cfg.max_correction_deg)
    sqrt_info = edge_information(source, target, candidate, cfg) if accepted else None
    accepted = accepted and sqrt_info is not None
    info.update(accepted=bool(accepted), overlap=overlap, rmse_m=rmse if np.isfinite(rmse) else None,
                condition=condition if np.isfinite(condition) else None,
                correction_m=correction_m, correction_deg=correction_deg,
                reason='accepted' if accepted else 'quality_gate')
    return ((candidate, sqrt_info) if accepted else None), info


def optimize_graph(poses, edges, cfg):
    """poses: [N,3] keyframe poses; edges: (i, j, measured, sqrt_info[3,3]). First pose fixed."""
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 2 or not edges:
        return poses
    first = poses[0].copy()

    def residual(values):
        current = np.vstack((first, values.reshape(-1, 3)))
        out = []
        for i, j, measured, sqrt_info in edges:
            error = relative(current[i], current[j]) - measured
            error[2] = wrap_angle(error[2])
            out.append(sqrt_info @ error)
        return np.concatenate(out)

    result = least_squares(residual, poses[1:].ravel(), loss='soft_l1',
                           f_scale=cfg.robust_scale, max_nfev=200)
    if not np.isfinite(result.x).all():
        return poses
    optimized = np.vstack((first, result.x.reshape(-1, 3)))
    optimized[:, 2] = wrap_angle(optimized[:, 2])
    return optimized


def make_learned_estimator(model, files, device, cfg):
    """Keyframe constraints measured by the trained BEV network instead of ICP.

    Used only inside the motion range seen in training; accepted only if the
    forward and reverse predictions cancel (no ground truth involved).
    """
    import torch

    cache = {}

    def bev(frame):
        if frame not in cache:
            with np.load(files[frame], allow_pickle=False) as archive:
                cache[frame] = torch.from_numpy(archive['bev'].astype(np.float32))
        return cache[frame]

    sqrt_info = np.diag(1.0 / np.array([cfg.learned_sigma_m, cfg.learned_sigma_m,
                                        math.radians(cfg.learned_sigma_deg)]))

    def estimate(frame_i, frame_k, initial, loop=False):
        info = dict(accepted=False)
        if (np.hypot(*initial[:2]) > cfg.learned_max_m
                or abs(math.degrees(initial[2])) > cfg.learned_max_deg):
            info['reason'] = 'outside_trained_range'
            return None, info
        with torch.no_grad():
            a, b = bev(frame_i)[None].to(device), bev(frame_k)[None].to(device)
            poses = model(torch.cat((a, b)), torch.cat((b, a)))['pose'].float().cpu().numpy()
        forward, backward = poses[0].astype(float), poses[1].astype(float)
        loop_error = compose(forward, backward)
        delta = relative(initial, forward)
        consistent = (np.hypot(*loop_error[:2]) <= cfg.learned_consistency_m
                      and abs(math.degrees(wrap_angle(loop_error[2]))) <= cfg.learned_consistency_deg)
        accepted = (consistent and np.hypot(*delta[:2]) <= cfg.max_correction_m
                    and abs(math.degrees(delta[2])) <= cfg.max_correction_deg)
        info.update(accepted=bool(accepted), reason='accepted' if accepted else 'consistency_or_gate',
                    consistency_m=float(np.hypot(*loop_error[:2])),
                    consistency_deg=abs(math.degrees(wrap_angle(loop_error[2]))))
        return ((forward, sqrt_info) if accepted else None), info

    return estimate


def run_keyframe_backend(increments, point_loader, cfg, edge_estimator=None):
    """Return raw frame poses, optimized frame poses, keyframe indices, diagnostics.

    ``edge_estimator(frame_i, frame_k, initial, loop)`` replaces ICP when given.
    """
    n = len(increments) + 1
    raw = [np.zeros(3)]
    for inc in increments:
        raw.append(compose(raw[-1], inc))
    raw = np.asarray(raw)

    keyframes = [0]                 # frame indices
    kf_poses = [np.zeros(3)]        # optimized keyframe poses
    kf_points = {0: point_loader(0)}
    edges, diagnostics = [], []
    odom_sigma = np.array([cfg.odom_sigma_m, cfg.odom_sigma_m, math.radians(cfg.odom_sigma_deg)])
    icp_sigma = np.array([cfg.icp_sigma_m, cfg.icp_sigma_m, math.radians(cfg.icp_sigma_deg)])

    for frame in range(1, n):
        last = keyframes[-1]
        since = relative(raw[last], raw[frame])
        is_last_frame = frame == n - 1
        if not (np.hypot(*since[:2]) >= cfg.keyframe_distance_m
                or abs(math.degrees(since[2])) >= cfg.keyframe_angle_deg
                or frame - last >= cfg.max_gap or is_last_frame):
            continue
        k = len(keyframes)
        keyframes.append(frame)
        kf_poses.append(compose(kf_poses[-1], since))
        kf_points[k] = point_loader(frame)
        gap = frame - last
        edges.append((k - 1, k, since, np.diag(1.0 / (odom_sigma * math.sqrt(gap)))))
        record = dict(keyframe=k, frame=frame, neighbor_edges=0, loop_edges=0, tried=0)
        def measure(i, initial, loop=False):
            if edge_estimator is not None:
                return edge_estimator(keyframes[i], keyframes[k], initial, loop)
            return register_pair(kf_points[k], kf_points[i], initial, cfg, loop=loop)

        for i in range(max(0, k - cfg.neighbors), k):
            initial = relative(kf_poses[i], kf_poses[k])
            result, info = measure(i, initial)
            record['tried'] += 1
            if result is not None:
                edges.append((i, k, result[0], result[1]))
                record['neighbor_edges'] += 1
        for i in range(0, max(0, k - cfg.neighbors)):
            initial = relative(kf_poses[i], kf_poses[k])
            if (np.hypot(*initial[:2]) > cfg.loop_radius_m
                    or abs(math.degrees(initial[2])) > cfg.loop_yaw_deg):
                continue
            result, info = measure(i, initial, loop=True)
            if result is not None:
                edges.append((i, k, result[0], result[1]))
                record['loop_edges'] += 1
        kf_poses = list(optimize_graph(kf_poses, edges, cfg))
        diagnostics.append(record)

    # Attach every frame to its preceding keyframe through frontend increments.
    optimized = np.zeros_like(raw)
    kf_iter = 0
    for frame in range(n):
        while kf_iter + 1 < len(keyframes) and keyframes[kf_iter + 1] <= frame:
            kf_iter += 1
        anchor = keyframes[kf_iter]
        optimized[frame] = compose(kf_poses[kf_iter], relative(raw[anchor], raw[frame]))
    return raw, optimized, keyframes, diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--edge-model', help='keypoint_bev checkpoint: measure keyframe edges with the network')
    parser.add_argument('--keyframe-distance', type=float)
    parser.add_argument('--keyframe-angle', type=float)
    args = parser.parse_args(argv)
    cfg = KeyframeConfig()
    if args.keyframe_distance:
        cfg.keyframe_distance_m = args.keyframe_distance
    if args.keyframe_angle:
        cfg.keyframe_angle_deg = args.keyframe_angle
    model = device = None
    if args.edge_model:
        import torch
        from .keypoint_bev_model import KeypointBEVOdometry
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(args.edge_model, map_location=device, weights_only=False)
        model = KeypointBEVOdometry(**checkpoint['model_config']).to(device).eval()
        model.load_state_dict(checkpoint['model_state'])
    point_cfg = BackendConfig(voxel_m=cfg.voxel_m, max_points=cfg.max_points)
    output = Path(args.output_dir)
    if output.exists():
        parser.error('output directory exists')
    with Path(args.predictions).open(encoding='utf-8-sig', newline='') as stream:
        predictions = list(csv.DictReader(stream))
    with Path(args.manifest).open(encoding='utf-8') as stream:
        manifests = {pair_key(r): r for r in (json.loads(l) for l in stream if l.strip())}
    output.mkdir(parents=True)
    summary = dict(config=asdict(cfg), edge_model=args.edge_model,
                   predictions=str(Path(args.predictions).resolve()),
                   manifest=str(Path(args.manifest).resolve()), segments=[])
    trajectories = []
    for segment, (sequence, chain) in enumerate(select_chains(predictions)):
        increments = np.array([[float(r[k]) for k in ('pred_forward_m', 'pred_left_m', 'pred_yaw_rad')]
                               for r in chain])
        matching = [manifests[pair_key(r)] for r in chain]
        files = [matching[0]['bev_0']] + [r['bev_1'] for r in matching]
        estimator = make_learned_estimator(model, files, device, cfg) if model is not None else None
        raw, corrected, keyframes, diagnostics = run_keyframe_backend(
            increments, lambda i: load_points(files[i], point_cfg), cfg, edge_estimator=estimator)
        target = [np.zeros(3)]
        for r in matching:   # ground truth only for scoring, after the backend finished
            target.append(compose(target[-1], [r['forward_m'], r['left_m'], r['yaw_rad']]))
        target = np.asarray(target)
        times = [chain[0]['timestamp_0']] + [r['timestamp_1'] for r in chain]
        for i, t in enumerate(times):
            entry = dict(sequence=sequence, segment=segment, timestamp=float(t), bev=files[i],
                         keyframe=int(i in keyframes))
            for name, values in (('raw', raw), ('optimized', corrected), ('target', target)):
                entry.update({f'{name}_{a}': float(v) for a, v in zip(('x_m', 'y_m', 'yaw_rad'), values[i])})
            trajectories.append(entry)
        summary['segments'].append(dict(
            sequence=sequence, segment=segment, pairs=len(chain), keyframes=len(keyframes),
            neighbor_edges=sum(d['neighbor_edges'] for d in diagnostics),
            neighbor_tried=sum(d['tried'] for d in diagnostics),
            loop_edges=sum(d['loop_edges'] for d in diagnostics),
            raw=metrics(raw, target), optimized=metrics(corrected, target)))
        (output / f'keyframes_{segment}.json').write_text(json.dumps(diagnostics, indent=1), encoding='utf-8')
    lines = ['# 关键帧位姿图后端（v2）评估', '',
             '参考轨迹：真值平面增量积分，仅起点对齐。真值不参与后端。', '',
             '| 序列 / 片段 | 帧对 | 关键帧 | 邻近约束 成功/尝试 | 回环约束 | 位置RMSE 前端→后端 (m) | 终点位置 (m) | 终点角度 (°) | 角度RMSE (°) |',
             '|---|---|---|---|---|---|---|---|---|']
    for s in summary['segments']:
        a, b = s['raw'], s['optimized']
        lines.append(f"| {s['sequence']} / {s['segment']} | {s['pairs']} | {s['keyframes']} | "
                     f"{s['neighbor_edges']}/{s['neighbor_tried']} | {s['loop_edges']} | "
                     f"{a['position_rmse_start_aligned_m']:.3f} → {b['position_rmse_start_aligned_m']:.3f} | "
                     f"{a['endpoint_translation_error_m']:.3f} → {b['endpoint_translation_error_m']:.3f} | "
                     f"{a['endpoint_yaw_error_deg']:.2f} → {b['endpoint_yaw_error_deg']:.2f} | "
                     f"{a['yaw_rmse_deg']:.2f} → {b['yaw_rmse_deg']:.2f} |")
    (output / '报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_csv(output / 'trajectories.csv', trajectories)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
