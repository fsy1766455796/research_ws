"""Appearance-based loop closure SLAM backend (v5).

Place recognition with an off-the-shelf pretrained image model (DINOv2,
facebook/dinov2-base, used frozen) and geometric verification with SIFT +
depth; the pose graph, session placement and scoring come from
``loop_slam``.  The backend never reads ground truth; ground truth is used
afterwards for scoring and for loop-precision diagnostics only.

Pipeline
1. Sessions = chains of frontend increments (``loop_slam.load_sessions``);
   keyframes every 0.25 m / 8 deg.
2. Every keyframe colour image -> DINOv2 global descriptor
   (L2-normalised CLS token concatenated with mean patch token).
3. Candidates: for each keyframe the ``top_k`` most similar earlier
   keyframes (other session, or same session >= ``min_separation_m`` of path
   apart), similarity >= ``min_similarity``.
4. Verification: SIFT matches (ratio test) between the two colour images,
   matched pixels back-projected with depth (sensor depth or DA3) into the
   robot frame [forward, left, up]; 2-point RANSAC for a planar rigid motion
   (heights must agree as well); accepted with enough inliers.
5. New sessions are placed by consensus of verified matches; all verified
   matches become loop edges of a robust pose graph; edges with large
   residuals are removed and the graph re-optimized.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .local_map_backend import compose, relative, write_csv
from .loop_slam import (Graph, LoopConfig, fit_se2, groundtruth_planar, load_sessions,
                        prune_outliers, rmse, start_aligned, transform_traj)
from .multichannel_bev import (OPENLORIS_COLOR_INTRINSICS, OPENLORIS_COLOR_SIZE,
                               OPENLORIS_T_BASE_COLOR, scale_intrinsics)
from .openloris_data import (_load_da3_depth, _load_sensor_depth, _nearest_entry,
                             _resolve_da3_prediction, resolve_sequence_dir, sequence_calibration)
from .openloris_geometry import load_timestamped_paths
from .se2 import wrap_angle


@dataclass
class AppearanceConfig:
    top_k: int = 3
    min_similarity: float = 0.55
    min_separation_m: float = 2.5
    sift_features: int = 2000
    ratio: float = 0.8
    min_depth_m: float = 0.3
    max_depth_m: float = 6.0
    ransac_iterations: int = 400
    inlier_m: float = 0.10
    height_m: float = 0.15
    min_inliers: int = 25
    min_inlier_ratio: float = 0.30
    max_loop_translation_m: float = 4.0
    # v6 additions (defaults keep v5 behaviour)
    consistency_window: int = 0        # >0: keep a loop only if another loop within +-window keyframes agrees
    consistency_m: float = 0.3
    consistency_deg: float = 5.0
    neighbor_edges: int = 0            # >0: metric SIFT+depth edges to the previous N keyframes of a session
    neighbor_sigma_m: float = 0.15
    neighbor_sigma_deg: float = 3.0
    estimate_scale: bool = False       # per-session scale of the frontend translation in the graph
    scale_prior_sigma: float = 0.3     # on log scale
    # v10: loop verification mode. 'rigid2d' = 3D-3D planar RANSAC (v5-v9);
    # 'pnp' = 3D points of frame a (its depth only) vs pixels of frame b, PnP RANSAC,
    # projected to the planar robot motion (same geometry as the v8 frontend).
    verify_mode: str = 'rigid2d'
    pnp_reproj_px: float = 3.0


# ------------------------------------------------------------ data access

class FrameSource:
    """Colour image and depth for a frame timestamp of one sequence."""

    def __init__(self, dataset_root, depth_source, da3_root=None):
        self.root, self.depth_source, self.da3_root = dataset_root, depth_source, da3_root
        self.index = {}

    def _sequence(self, sequence):
        if sequence not in self.index:
            d = resolve_sequence_dir(self.root, sequence)
            colors = load_timestamped_paths(d / 'color.txt')
            depths = load_timestamped_paths(d / 'aligned_depth.txt') if self.depth_source == 'aligned_depth' else None
            self.index[sequence] = (d, {round(c.timestamp, 6): c for c in colors}, depths)
        return self.index[sequence]

    def calibration(self, sequence):
        """(fx, fy, cx, cy), (w, h), T_base_color read from the sequence's own calibration files."""
        if not hasattr(self, '_calib'):
            self._calib = {}
        if sequence not in self._calib:
            self._calib[sequence] = sequence_calibration(self.root, sequence)
        return self._calib[sequence]

    def color_path(self, sequence, timestamp):
        d, colors, _ = self._sequence(sequence)
        return d / colors[round(float(timestamp), 6)].path

    def depth(self, sequence, timestamp):
        d, colors, depths = self._sequence(sequence)
        color = colors[round(float(timestamp), 6)]
        if self.depth_source == 'aligned_depth':
            entry = _nearest_entry(depths, color.timestamp, max_difference_sec=0.04)
            return None if entry is None else _load_sensor_depth(d / entry.path)
        found = _resolve_da3_prediction(self.da3_root, sequence, color.path, color.timestamp)
        return None if found is None else _load_da3_depth(found)


# ------------------------------------------------------------ descriptors

class Describer:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device).eval()
        if self.device == 'cuda':
            self.model = self.model.half()

    def __call__(self, paths, batch=16):
        from PIL import Image
        out = []
        for i in range(0, len(paths), batch):
            images = [Image.open(p).convert('RGB') for p in paths[i:i + batch]]
            inputs = self.processor(images=images, return_tensors='pt').to(self.device)
            if self.device == 'cuda':
                inputs['pixel_values'] = inputs['pixel_values'].half()
            with self.torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state.float()
            cls = self.torch.nn.functional.normalize(hidden[:, 0], dim=1)
            patch = self.torch.nn.functional.normalize(hidden[:, 1:].mean(1), dim=1)
            out.append(self.torch.nn.functional.normalize(self.torch.cat([cls, patch], 1), dim=1).cpu().numpy())
        return np.concatenate(out)


# ------------------------------------------------------------ verification

class Verifier:
    def __init__(self, source, cfg):
        import cv2
        self.cv2, self.source, self.cfg = cv2, source, cfg
        self.sift = cv2.SIFT_create(cfg.sift_features)
        self.cache = {}

    def features(self, sequence, timestamp):
        key = (sequence, round(float(timestamp), 6))
        if key not in self.cache:
            image = self.cv2.imread(str(self.source.color_path(sequence, timestamp)), self.cv2.IMREAD_GRAYSCALE)
            kps, desc = self.sift.detectAndCompute(image, None)
            depth = self.source.depth(sequence, timestamp)
            pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2)
            self.calib = self.source.calibration(sequence)
            xyz, optical = self._backproject(pts, depth, image.shape, with_optical=True)
            self.cache[key] = (desc, xyz, pts, optical, image.shape, self.calib)
        return self.cache[key]

    def _backproject(self, pixels, depth, image_shape, with_optical=False):
        out = np.full((len(pixels), 3), np.nan)
        out_optical = np.full((len(pixels), 3), np.nan)
        if depth is None or not len(pixels):
            return (out, out_optical) if with_optical else out
        h, w = depth.shape
        sx, sy = w / image_shape[1], h / image_shape[0]
        u = np.clip(np.round(pixels[:, 0] * sx).astype(int), 0, w - 1)
        v = np.clip(np.round(pixels[:, 1] * sy).astype(int), 0, h - 1)
        z = depth[v, u].astype(np.float64)
        intr, size, t_bc = self.calib
        fx, fy, cx, cy = scale_intrinsics(intr, size, (w, h))
        ok = np.isfinite(z) & (z >= self.cfg.min_depth_m) & (z <= self.cfg.max_depth_m)
        optical = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        robot = optical @ t_bc[:3, :3].T + t_bc[:3, 3]
        out[ok] = robot[ok]
        if with_optical:
            # optical coordinates at full image resolution (pixel scale does not change metric z)
            out_optical[ok] = optical[ok]
            return out, out_optical
        return out

    def verify(self, a, b):
        """a, b: (sequence, timestamp). Return (pose of b in a, info) or (None, info)."""
        c = self.cfg
        desc_a, xyz_a, _, opt_a, shape_a, calib_a = self.features(*a)
        desc_b, xyz_b, pix_b, _, shape_b, calib_b = self.features(*b)
        info = dict(matches=0, inliers=0)
        if c.verify_mode == 'pnp':
            return self._verify_pnp(desc_a, opt_a, calib_a, desc_b, pix_b, shape_b, calib_b, info)
        if desc_a is None or desc_b is None or len(desc_a) < 10 or len(desc_b) < 10:
            return None, info
        knn = self.cv2.BFMatcher(self.cv2.NORM_L2).knnMatch(desc_b, desc_a, k=2)
        pairs = [(m.queryIdx, m.trainIdx) for m, n in (x for x in knn if len(x) == 2)
                 if m.distance < c.ratio * n.distance]
        pairs = [(i, j) for i, j in pairs if np.isfinite(xyz_b[i]).all() and np.isfinite(xyz_a[j]).all()]
        info['matches'] = len(pairs)
        if len(pairs) < c.min_inliers:
            return None, info
        src = np.array([xyz_b[i] for i, _ in pairs])
        dst = np.array([xyz_a[j] for _, j in pairs])
        rng = np.random.default_rng(0)
        best, best_mask = None, None
        for _ in range(c.ransac_iterations):
            i, j = rng.choice(len(pairs), 2, replace=False)
            if np.hypot(*(src[i, :2] - src[j, :2])) < 0.2:
                continue
            pose = fit_se2(src[[i, j], :2], dst[[i, j], :2])
            mask = self._inliers(pose, src, dst)
            if best_mask is None or mask.sum() > best_mask.sum():
                best, best_mask = pose, mask
        if best_mask is None or best_mask.sum() < 3:
            return None, info
        pose = fit_se2(src[best_mask, :2], dst[best_mask, :2])
        mask = self._inliers(pose, src, dst)
        if mask.sum() >= 3:
            pose = fit_se2(src[mask, :2], dst[mask, :2])
        inliers = int(mask.sum())
        info.update(inliers=inliers, ratio=inliers / len(pairs))
        ok = (inliers >= c.min_inliers and inliers / len(pairs) >= c.min_inlier_ratio
              and np.hypot(*pose[:2]) <= c.max_loop_translation_m)
        info['accepted'] = bool(ok)
        return (pose if ok else None), info

    def _verify_pnp(self, desc_a, opt_a, calib_a, desc_b, pix_b, shape_b, calib_b, info):
        c, cv2 = self.cfg, self.cv2
        if desc_a is None or desc_b is None or len(desc_a) < 10 or len(desc_b) < 10:
            return None, info
        knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc_b, desc_a, k=2)
        pairs = [(m.queryIdx, m.trainIdx) for m, n in (x for x in knn if len(x) == 2)
                 if m.distance < c.ratio * n.distance]
        pairs = [(i, j) for i, j in pairs if np.isfinite(opt_a[j]).all()]
        info['matches'] = len(pairs)
        if len(pairs) < c.min_inliers:
            return None, info
        obj = np.array([opt_a[j] for _, j in pairs], dtype=np.float64)
        img = np.array([pix_b[i] for i, _ in pairs], dtype=np.float64)
        intr, size, _ = calib_b
        fx, fy, cx, cy = scale_intrinsics(intr, size, (shape_b[1], shape_b[0]))
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        found, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, k, None, reprojectionError=c.pnp_reproj_px,
                                                    iterationsCount=c.ransac_iterations, confidence=0.999,
                                                    flags=cv2.SOLVEPNP_EPNP)
        if not found or inl is None or len(inl) < 4:
            return None, info
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], k, None, rvec, tvec)
        r, _ = cv2.Rodrigues(rvec)
        t_c = np.eye(4)
        t_c[:3, :3], t_c[:3, 3] = r, tvec.ravel()
        t_bc = calib_a[2]
        t_b = t_bc @ np.linalg.inv(t_c) @ np.linalg.inv(t_bc)
        pose = np.array([t_b[0, 3], t_b[1, 3], math.atan2(t_b[1, 0], t_b[0, 0])])
        inliers = len(inl)
        info.update(inliers=inliers, ratio=inliers / len(pairs), dz=float(t_b[2, 3]))
        ok = (inliers >= c.min_inliers and inliers / len(pairs) >= c.min_inlier_ratio
              and np.hypot(*pose[:2]) <= c.max_loop_translation_m and abs(t_b[2, 3]) <= c.height_m)
        info['accepted'] = bool(ok)
        return (pose if ok else None), info

    def _inliers(self, pose, src, dst):
        cs, sn = math.cos(pose[2]), math.sin(pose[2])
        moved = np.column_stack((cs * src[:, 0] - sn * src[:, 1] + pose[0],
                                 sn * src[:, 0] + cs * src[:, 1] + pose[1]))
        return ((np.hypot(*(moved - dst[:, :2]).T) <= self.cfg.inlier_m)
                & (np.abs(src[:, 2] - dst[:, 2]) <= self.cfg.height_m))


# ------------------------------------------------------------ pipeline

def candidate_pairs(sessions, descriptors, cfg):
    nodes = [(s, k) for s, sess in enumerate(sessions) for k in range(len(sess.keyframes))]
    desc = np.concatenate([descriptors[s] for s in range(len(sessions))])
    order = {s: sessions[s].timestamps[0] for s in range(len(sessions))}
    sims = desc @ desc.T
    pairs = []
    for j, (s, kb) in enumerate(nodes):
        allowed = []
        for i, (t, ka) in enumerate(nodes):
            if t == s:
                sess = sessions[s]
                if (sess.path[sess.keyframes[kb]] - sess.path[sess.keyframes[ka]]) < cfg.min_separation_m:
                    continue
            elif order[t] > order[s]:
                continue
            if sims[j, i] >= cfg.min_similarity:
                allowed.append((sims[j, i], i))
        for sim, i in sorted(allowed, reverse=True)[:cfg.top_k]:
            pairs.append((i, j, float(sim)))
    return nodes, pairs


def run(sessions, frames, describer, loop_cfg, app_cfg, log):
    descriptors = {s: describer([frames.color_path(sess.sequence, sess.timestamps[f]) for f in sess.keyframes])
                   for s, sess in enumerate(sessions)}
    nodes, pairs = candidate_pairs(sessions, descriptors, app_cfg)
    verifier = Verifier(frames, app_cfg)
    verified = []
    for i, j, sim in pairs:
        (t, ka), (s, kb) = nodes[i], nodes[j]
        a = (sessions[t].sequence, sessions[t].timestamps[sessions[t].keyframes[ka]])
        b = (sessions[s].sequence, sessions[s].timestamps[sessions[s].keyframes[kb]])
        pose, info = verifier.verify(a, b)
        if pose is not None:
            verified.append(dict(a=(t, ka), b=(s, kb), pose=pose, sim=sim, **info))
    log.append(dict(stage='verify', candidates=len(pairs), verified=len(verified)))
    all_verified = verified
    verified = consistent_loops(verified, sessions, app_cfg)
    log.append(dict(stage='consistency', kept=len(verified), dropped=len(all_verified) - len(verified)))
    neighbors = neighbor_measurements(sessions, verifier, app_cfg)
    log.append(dict(stage='neighbors', edges=len(neighbors)))

    graph = Graph(sessions, loop_cfg)
    order = sorted(range(len(sessions)), key=lambda s: sessions[s].timestamps[0])
    graph.place(order[0], np.zeros(3))
    for s in order[1:]:
        hypotheses = []
        for v in verified:
            for (t, ka), (u, kb), pose in ((v['a'], v['b'], v['pose']),
                                          (v['b'], v['a'], relative(v['pose'], np.zeros(3)))):
                if u == s and graph.placed[t] and t != s:
                    world_b = compose(graph.poses[graph.node(t, ka)], pose)
                    start = compose(world_b, relative(sessions[s].raw[sessions[s].keyframes[kb]], np.zeros(3)))
                    hypotheses.append(dict(start=start, inliers=v['inliers']))
        best, support = None, []
        for h in hypotheses:
            inl = [g for g in hypotheses if np.hypot(*(g['start'][:2] - h['start'][:2])) <= loop_cfg.consensus_m
                   and abs(math.degrees(wrap_angle(g['start'][2] - h['start'][2]))) <= loop_cfg.consensus_deg]
            if len(inl) > len(support):
                best, support = h, inl
        placed = best is not None and len(support) >= loop_cfg.min_consensus
        log.append(dict(stage='place', session=sessions[s].name, hypotheses=len(hypotheses),
                        support=len(support), placed=placed))
        if placed:
            graph.place(s, best['start'])
    for v in verified:
        i, j = graph.node(*v['a']), graph.node(*v['b'])
        if graph.placed[v['a'][0]] and graph.placed[v['b'][0]]:
            graph.loop_edges[(i, j)] = (v['pose'], dict(kind='appearance', sim=v['sim'], inliers=v['inliers']))
    use_scale_graph = app_cfg.estimate_scale or app_cfg.neighbor_edges > 0
    if use_scale_graph and not app_cfg.estimate_scale:
        app_cfg.scale_prior_sigma = 1e-6     # scale fixed at 1

    def optimize():
        if use_scale_graph:
            optimize_with_scale(graph, neighbors, app_cfg)
        else:
            graph.optimize()

    optimize()
    for _ in range(2):
        if not prune_outliers(graph, log):
            break
        optimize()
    if hasattr(graph, 'session_scale'):
        log.append(dict(stage='scale', **graph.session_scale))
    for v in all_verified:
        v['consistent'] = any(v is w for w in verified)
    return graph, all_verified


def consistent_loops(verified, sessions, cfg):
    """Keep loops supported by another loop between nearby keyframes of the same session pair."""
    if cfg.consistency_window <= 0:
        return verified
    def odo(s, k0, k1):
        sess = sessions[s]
        return relative(sess.raw[sess.keyframes[k0]], sess.raw[sess.keyframes[k1]])
    kept = []
    for v in verified:
        (t, ka), (s, kb) = v['a'], v['b']
        for w in verified:
            if w is v:
                continue
            (t2, ka2), (s2, kb2) = w['a'], w['b']
            if (t2, s2) != (t, s) or abs(ka2 - ka) > cfg.consistency_window or abs(kb2 - kb) > cfg.consistency_window:
                continue
            predicted = compose(compose(relative(odo(t, ka, ka2), np.zeros(3)), v['pose']), odo(s, kb, kb2))
            e = relative(predicted, w['pose'])
            if np.hypot(*e[:2]) <= cfg.consistency_m and abs(math.degrees(wrap_angle(e[2]))) <= cfg.consistency_deg:
                kept.append(v)
                break
    return kept


def neighbor_measurements(sessions, verifier, cfg):
    edges = []
    if cfg.neighbor_edges <= 0:
        return edges
    for s, sess in enumerate(sessions):
        for kb in range(1, len(sess.keyframes)):
            for ka in range(max(0, kb - cfg.neighbor_edges), kb):
                a = (sess.sequence, sess.timestamps[sess.keyframes[ka]])
                b = (sess.sequence, sess.timestamps[sess.keyframes[kb]])
                pose, info = verifier.verify(a, b)
                if pose is not None:
                    edges.append(dict(a=(s, ka), b=(s, kb), pose=pose, **info))
    return edges


def optimize_with_scale(graph, extra_edges, app_cfg):
    """Pose graph with one log-scale per session multiplying frontend translations."""
    from scipy.optimize import least_squares
    c = graph.cfg
    active = [i for i, (s, _) in enumerate(graph.nodes) if graph.placed[s]]
    remap = {old: new for new, old in enumerate(active)}
    sessions_active = sorted({graph.nodes[i][0] for i in active})
    sidx = {s: n for n, s in enumerate(sessions_active)}
    odom_sigma = np.array([c.odom_sigma_m, c.odom_sigma_m, math.radians(c.odom_sigma_deg)])
    odo = []
    for s in sessions_active:
        sess = graph.sessions[s]
        for k in range(1, len(sess.keyframes)):
            a, b = sess.keyframes[k - 1], sess.keyframes[k]
            odo.append((remap[graph.node(s, k - 1)], remap[graph.node(s, k)], sidx[s],
                        relative(sess.raw[a], sess.raw[b]), np.diag(1.0 / (odom_sigma * math.sqrt(b - a)))))
    loop_info = graph.loop_info()
    meas = [(remap[i], remap[j], m, loop_info) for (i, j), (m, _) in graph.loop_edges.items()]
    n_info = np.diag(1.0 / np.array([app_cfg.neighbor_sigma_m, app_cfg.neighbor_sigma_m,
                                     math.radians(app_cfg.neighbor_sigma_deg)]))
    meas += [(remap[graph.node(*e['a'])], remap[graph.node(*e['b'])], e['pose'], n_info) for e in extra_edges
             if graph.placed[e['a'][0]]]
    poses = graph.poses[active].copy()
    first = poses[0].copy()
    n = len(poses)

    def unpack(x):
        return np.vstack((first, x[:3 * (n - 1)].reshape(-1, 3))), x[3 * (n - 1):]

    def residual(x):
        p, ls = unpack(x)
        out = []
        for i, j, s, m, w in odo:
            scale = math.exp(ls[s])
            e = relative(p[i], p[j]) - np.array([scale * m[0], scale * m[1], m[2]])
            e[2] = wrap_angle(e[2])
            out.append(w @ e)
        for i, j, m, w in meas:
            e = relative(p[i], p[j]) - m
            e[2] = wrap_angle(e[2])
            out.append(w @ e)
        out.append(ls / app_cfg.scale_prior_sigma)
        return np.concatenate(out)

    x0 = np.concatenate((poses[1:].ravel(), np.zeros(len(sessions_active))))
    result = least_squares(residual, x0, loss='soft_l1', f_scale=c.robust_scale, max_nfev=300)
    p, ls = unpack(result.x)
    p[:, 2] = wrap_angle(p[:, 2])
    graph.poses[active] = p
    graph.session_scale = {graph.sessions[s].name: float(math.exp(ls[sidx[s]])) for s in sessions_active}
    # frame poses must use the scaled frontend increments
    graph.scale_of = {s: math.exp(ls[sidx[s]]) for s in sessions_active}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs=2, action='append', required=True, metavar=('PREDICTIONS', 'MANIFEST'))
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--depth-source', choices=('aligned_depth', 'da3'), required=True)
    parser.add_argument('--da3-root')
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--set', nargs=2, action='append', default=[], metavar=('FIELD', 'VALUE'),
                        help='override an AppearanceConfig or LoopConfig field')
    parser.add_argument('--score-sessions', default='', help='comma list of sequences to report (default all)')
    args = parser.parse_args(argv)
    loop_cfg, app_cfg = LoopConfig(build_submaps=False), AppearanceConfig()
    for field, value in args.set:
        target = app_cfg if hasattr(app_cfg, field) else loop_cfg
        kind = type(getattr(target, field))
        setattr(target, field, (value.lower() in ('1', 'true')) if kind is bool else kind(value))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sessions = load_sessions(args.input, loop_cfg)
    frames = FrameSource(args.dataset_root, args.depth_source, args.da3_root)
    log = []
    graph, verified = run(sessions, frames, Describer(args.model_dir), loop_cfg, app_cfg, log)

    # ---------------- scoring (ground truth only from here)
    gts = [groundtruth_planar(args.dataset_root, s.sequence, s.timestamps) for s in sessions]
    for v in verified:     # loop precision diagnostic
        (t, ka), (s, kb) = v['a'], v['b']
        rel = relative(gts[t][sessions[t].keyframes[ka]], gts[s][sessions[s].keyframes[kb]])
        e = relative(rel, v['pose'])
        v['error_m'], v['error_deg'] = float(np.hypot(*e[:2])), abs(math.degrees(wrap_angle(e[2])))
    kept = {(graph.nodes[i], graph.nodes[j]) for (i, j) in graph.loop_edges}
    for v in verified:
        v['kept'] = (v['a'], v['b']) in kept
    est = {i: graph.frame_poses(i) for i in range(len(sessions)) if graph.placed[i]}
    placed = list(est)
    all_est = np.concatenate([est[i] for i in placed])
    all_gt = np.concatenate([gts[i] for i in placed])
    world = fit_se2(all_est[:, :2], all_gt[:, :2])
    slam_world = {i: transform_traj(world, est[i]) for i in placed}
    front_world = {i: transform_traj(gts[i][0], np.array([relative(s.raw[0], p) for p in s.raw]))
                   for i, s in enumerate(sessions)}
    good = lambda v: v['error_m'] < 0.3 and v['error_deg'] < 5
    report = dict(appearance=asdict(app_cfg), loop=asdict(loop_cfg), inputs=args.input, log=log,
                  loops=dict(verified=len(verified), verified_correct=sum(good(v) for v in verified),
                             kept=sum(v['kept'] for v in verified),
                             kept_correct=sum(v['kept'] and good(v) for v in verified)),
                  joint=dict(sessions_placed=len(placed), sessions_total=len(sessions),
                             slam_ate_single_alignment_m=rmse(np.concatenate([slam_world[i] for i in placed]), all_gt),
                             frontend_ate_true_start_m=rmse(np.concatenate(list(front_world.values())),
                                                            np.concatenate(gts))),
                  sessions=[])
    for i, s in enumerate(sessions):
        e = dict(name=s.name, frames=len(s.raw), keyframes=len(s.keyframes), path_m=float(s.path[-1]),
                 placed=graph.placed[i], frontend=start_aligned(s.raw, gts[i]))
        if graph.placed[i]:
            e['slam'] = start_aligned(est[i], gts[i])
            e['slam_world_ate_m'] = rmse(slam_world[i], gts[i])
        e['frontend_world_ate_m'] = rmse(front_world[i], gts[i])
        report['sessions'].append(e)
    rows = []
    for i, s in enumerate(sessions):
        kfs = set(s.keyframes)
        for f in range(len(s.raw)):
            row = dict(sequence=s.sequence, segment=i, session=s.name, timestamp=float(s.timestamps[f]),
                       bev=s.files[f], keyframe=int(f in kfs))
            for name, traj in (('raw', front_world[i]), ('optimized', slam_world.get(i)), ('target', gts[i])):
                vals = traj[f] if traj is not None else [np.nan] * 3
                row.update({f'{name}_{a}': float(v) for a, v in zip(('x_m', 'y_m', 'yaw_rad'), vals)})
            rows.append(row)
    write_csv(output / 'trajectories.csv', rows)
    with (output / 'loops.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        w = csv.writer(stream)
        w.writerow(['a', 'b', 'sim', 'matches', 'inliers', 'ratio', 'error_m', 'error_deg', 'kept'])
        for v in verified:
            w.writerow([v['a'], v['b'], round(v['sim'], 3), v['matches'], v['inliers'], round(v['ratio'], 3),
                        round(v['error_m'], 3), round(v['error_deg'], 2), v['kept']])
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                                         encoding='utf-8')
    L = report['loops']
    lines = ['# 外观回环 SLAM（v5）评估', '',
             '后端不用真值；真值只用于事后评分和回环正确率诊断。“世界系”指所有段放在同一张地图里，只做一次整体刚体对齐。前端基线把每段放在真实起点（它本身做不到）。', '',
             f"回环：通过几何验证 {L['verified']} 条（其中正确 {L['verified_correct']}），剔除离群后保留 {L['kept']} 条（正确 {L['kept_correct']}）。正确 = 与真值相差 < 0.3 m 且 < 5°。", '',
             '| 段 | 路程 (m) | 已接入地图 | 前端 ATE 起点对齐 | SLAM ATE 起点对齐 | 前端 世界系 ATE | SLAM 世界系 ATE | SLAM 终点角 (°) |',
             '|---|---|---|---|---|---|---|---|']
    for e in report['sessions']:
        g = e.get('slam', {})
        lines.append(f"| {e['name']} | {e['path_m']:.1f} | {'是' if e['placed'] else '否'} | {e['frontend']['ate_m']:.3f} | "
                     f"{g.get('ate_m', float('nan')):.3f} | {e['frontend_world_ate_m']:.3f} | "
                     f"{e.get('slam_world_ate_m', float('nan')):.3f} | {g.get('endpoint_yaw_deg', float('nan')):.1f} |")
    j = report['joint']
    lines += ['', f"整体：SLAM 世界系 ATE {j['slam_ate_single_alignment_m']:.3f} m（{j['sessions_placed']}/{j['sessions_total']} 段接入）；"
              f"前端（每段用真实起点）{j['frontend_ate_true_start_m']:.3f} m。"]
    (output / '报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(json.dumps(log, ensure_ascii=False))


if __name__ == '__main__':
    main()
