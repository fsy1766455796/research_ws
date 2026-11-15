"""Loop-closure SLAM backend (v4): submaps, place recognition, multi-session graph.

STATUS 2026-11-08 (validation office1-5, measured-depth new model): NOT usable.
Matcher self-test recovers known transforms exactly, but on real revisits the
correlative BEV matching is not discriminative: of 260 random keyframe pairs
(25 with true view overlap > 0.4) only 3 were solved within 0.5 m / 8 deg, and
no score/ambiguity threshold gives a precise accept set.  Forward camera,
revisits mostly in the opposite heading, repetitive office structure.  Place
recognition needs appearance (e.g. image features), not BEV geometry alone.

No ground truth is used by the backend.  Ground truth (OpenLORIS base_link in
the shared mocap frame) is read only afterwards for scoring and plots.

Why a new backend: in v2/v3 the neighbour edges (ICP or the same network)
were not more accurate than the frontend and carried no independent
information.  The only strong independent cue is a revisit.  Here:

1. Every chain of frontend increments is a *session* (a sequence, or a
   segment when the chain breaks).  Keyframes every 0.25 m / 8 deg.
2. Each keyframe gets a submap: occupied BEV cells of the frames within
   +-1 m of travelled path, placed with frontend odometry.  A single forward
   camera view is narrow; revisits often happen facing the other way.
3. Matching is correlative: for every yaw on a grid the score of all
   translations in a window is computed at once by FFT correlation against a
   Gaussian likelihood field of the target submap; the best coarse solution
   is refined on a finer grid.  Global (360 deg) search needs no initial
   guess.  A match is accepted only if its score is high, it is clearly better
   than the best alternative pose (ambiguity ratio) and the point overlap is
   large.
4. Place recognition places sessions relative to each other: keyframe pairs
   across sessions are matched globally and each accepted match votes for the
   start pose of the new session; the largest consistent cluster wins.
5. Loop edges: keyframe pairs far apart along the path (or in different
   sessions) whose current estimates are close are matched locally; the graph
   (frontend odometry edges + loop edges, robust loss, first keyframe fixed)
   is optimized; loop edges with large residuals are removed and the graph
   re-optimized.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .keyframe_backend import optimize_graph
from .local_map_backend import compose, pair_key, relative, select_chains, write_csv
from .se2 import wrap_angle


@dataclass
class LoopConfig:
    keyframe_distance_m: float = 0.25
    keyframe_angle_deg: float = 8.0
    max_gap: int = 10
    submap_path_m: float = 1.0
    submap_stride: int = 2
    max_range_m: float = 5.0
    point_channels: str = '7,8'
    build_submaps: bool = True
    min_points: int = 150
    coarse_resolution_m: float = 0.10
    coarse_sigma_m: float = 0.12
    coarse_yaw_step_deg: float = 2.0
    fine_resolution_m: float = 0.05
    fine_sigma_m: float = 0.06
    fine_yaw_step_deg: float = 0.5
    min_score: float = 0.50
    max_ambiguity: float = 0.90
    min_overlap: float = 0.45
    overlap_radius_m: float = 0.10
    ambiguity_distance_m: float = 0.5
    ambiguity_angle_deg: float = 20.0
    loop_min_separation_m: float = 2.5
    loop_radius_m: float = 1.0
    loop_radius_per_m: float = 0.08
    loop_candidates_per_keyframe: int = 3
    loop_window_m: float = 0.6
    place_stride: int = 2
    place_window_m: float = 2.5
    consensus_m: float = 0.6
    consensus_deg: float = 10.0
    min_consensus: int = 2
    odom_sigma_m: float = 0.02
    odom_sigma_deg: float = 0.6
    loop_sigma_m: float = 0.05
    loop_sigma_deg: float = 1.0
    robust_scale: float = 2.0
    outlier_chi: float = 3.0
    rounds: int = 2


_POINTS = {}


def frame_points(path, channels=(7, 8)):
    """Cell centres of a BEV frame where any of ``channels`` is set.

    Default: height bins 0.75-1.25 m and 1.25-1.8 m (walls, cabinets, desk
    tops).  The occupancy channel also contains floor returns, which form large
    textureless blobs that make correlative matching ambiguous.
    """
    key = (path, tuple(channels))
    if key not in _POINTS:
        meta = json.loads((Path(path).parent / 'metadata.json').read_text(encoding='utf-8'))['bev_config']
        with np.load(path, allow_pickle=False) as archive:
            bev = archive['bev']
        mask = np.zeros(bev.shape[1:], bool)
        for c in channels:
            mask |= bev[c] > 0.5
        rows, cols = np.nonzero(mask)
        res = float(meta['resolution_m'])
        _POINTS[key] = np.column_stack((meta['forward_min_m'] + (rows + 0.5) * res,
                                        meta['left_min_m'] + (cols + 0.5) * res))
    return _POINTS[key]


def inverse_pose(p):
    return relative(p, np.zeros(3))


def rotate(points, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.array([[c, s], [-s, c]])


def apply(pose, points):
    return rotate(points, pose[2]) + pose[:2]


# ---------------------------------------------------------------- sessions

class Session:
    def __init__(self, name, sequence, increments, files, timestamps, cfg):
        self.name, self.sequence = name, sequence
        self.files, self.timestamps = files, np.asarray(timestamps, dtype=float)
        raw = [np.zeros(3)]
        for inc in increments:
            raw.append(compose(raw[-1], inc))
        self.raw = np.asarray(raw)
        steps = np.linalg.norm(np.diff(self.raw[:, :2], axis=0), axis=1)
        self.path = np.concatenate(([0.0], np.cumsum(steps)))
        self.keyframes = [0]
        for f in range(1, len(self.raw)):
            since = relative(self.raw[self.keyframes[-1]], self.raw[f])
            if (np.hypot(*since[:2]) >= cfg.keyframe_distance_m
                    or abs(math.degrees(since[2])) >= cfg.keyframe_angle_deg
                    or f - self.keyframes[-1] >= cfg.max_gap or f == len(self.raw) - 1):
                self.keyframes.append(f)
        self.submaps = [self._submap(k, cfg) for k in self.keyframes] if cfg.build_submaps else []

    def _submap(self, k, cfg):
        near = np.nonzero(np.abs(self.path - self.path[k]) <= cfg.submap_path_m)[0]
        frames = sorted(set(near[::cfg.submap_stride].tolist()) | {k})
        clouds = []
        for f in frames:
            pts = frame_points(self.files[f], tuple(int(c) for c in cfg.point_channels.split(',')))
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) <= cfg.max_range_m]
            clouds.append(apply(relative(self.raw[k], self.raw[f]), pts))
        pts = np.concatenate(clouds)
        keys, counts = np.unique(np.floor(pts / cfg.fine_resolution_m).astype(np.int64),
                                 axis=0, return_counts=True)
        if len(frames) >= 4:
            keys = keys[counts >= 2]
        return (keys + 0.5) * cfg.fine_resolution_m


# ---------------------------------------------------------------- matching

class Field:
    """Gaussian likelihood field of a point set, padded for a search window."""

    def __init__(self, points, resolution, sigma, window_m):
        self.res = resolution
        self.pad = int(math.ceil(window_m / resolution)) + 2
        self.lo = points.min(0) - (self.pad + 1) * resolution
        idx = np.floor((points - self.lo) / resolution).astype(int)
        shape = idx.max(0) + self.pad + 2
        occ = np.zeros(shape, bool)
        occ[idx[:, 0], idx[:, 1]] = True
        dist = ndimage.distance_transform_edt(~occ) * resolution
        self.field = np.exp(-0.5 * (dist / sigma) ** 2)
        self.size = tuple(2 * s for s in self.field.shape)
        self.spectrum = np.fft.rfft2(self.field, self.size)

    def scores(self, points, window_cells):
        """Score map over integer cell offsets [-w, w]^2 for already-placed points."""
        shape = self.field.shape
        idx = np.floor((points - self.lo) / self.res).astype(int)
        keep = (idx >= 0).all(1) & (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1])
        hist = np.zeros(shape)
        np.add.at(hist, (idx[keep, 0], idx[keep, 1]), 1.0)
        conv = np.fft.irfft2(self.spectrum * np.fft.rfft2(hist[::-1, ::-1], self.size), self.size)
        cx, cy = shape[0] - 1, shape[1] - 1
        w = window_cells
        return conv[cx - w:cx + w + 1, cy - w:cy + w + 1] / max(len(points), 1)


def correlative_search(source, field, initial, yaw_offsets, window_m):
    """Return list of (score, pose, score_map, yaw) for each yaw offset."""
    w = int(round(window_m / field.res))
    out = []
    for dyaw in yaw_offsets:
        yaw = initial[2] + dyaw
        placed = rotate(source, yaw) + initial[:2]
        smap = field.scores(placed, w)
        i, j = np.unravel_index(np.argmax(smap), smap.shape)
        pose = np.array([initial[0] + (i - w) * field.res, initial[1] + (j - w) * field.res, yaw])
        out.append((float(smap[i, j]), pose, smap, yaw))
    return out


class Matcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = {}

    def fields(self, key, points, window_m):
        k = (key, round(window_m, 3))
        if k not in self.cache:
            c = self.cfg
            self.cache[k] = (Field(points, c.coarse_resolution_m, c.coarse_sigma_m, window_m),
                             Field(points, c.fine_resolution_m, c.fine_sigma_m,
                                   2 * c.coarse_resolution_m))
        return self.cache[k]

    def match(self, source, target, target_key, initial, yaw_range_deg, window_m):
        """Pose of the source submap frame in the target submap frame, or None."""
        c = self.cfg
        info = dict(accepted=False)
        if len(source) < c.min_points or len(target) < c.min_points:
            info['reason'] = 'few_points'
            return None, info
        coarse, fine = self.fields(target_key, target, window_m)
        if yaw_range_deg >= 180:
            offsets = np.radians(np.arange(-180, 180, c.coarse_yaw_step_deg))
        else:
            offsets = np.radians(np.arange(-yaw_range_deg, yaw_range_deg + 1e-9, c.coarse_yaw_step_deg))
        results = correlative_search(source, coarse, initial, offsets, window_m)
        best = max(range(len(results)), key=lambda r: results[r][0])
        best_score, best_pose, best_map, _ = results[best]
        # ambiguity: best alternative far from the winner (other yaw or other place)
        alternative = 0.0
        for score, pose, smap, _ in results:
            if abs(math.degrees(wrap_angle(pose[2] - best_pose[2]))) > c.ambiguity_angle_deg:
                alternative = max(alternative, score)
        w = (best_map.shape[0] - 1) // 2
        ii, jj = np.meshgrid(np.arange(best_map.shape[0]), np.arange(best_map.shape[1]), indexing='ij')
        bi, bj = np.unravel_index(np.argmax(best_map), best_map.shape)
        far = np.hypot(ii - bi, jj - bj) * coarse.res > c.ambiguity_distance_m
        if far.any():
            alternative = max(alternative, float(best_map[far].max()))
        # refine
        fine_offsets = np.radians(np.arange(-c.coarse_yaw_step_deg, c.coarse_yaw_step_deg + 1e-9,
                                            c.fine_yaw_step_deg))
        refined = correlative_search(source, fine, best_pose, fine_offsets, 2 * c.coarse_resolution_m)
        score, pose, _, _ = max(refined, key=lambda r: r[0])
        d, _ = cKDTree(target).query(apply(pose, source))
        overlap = float((d <= c.overlap_radius_m).mean())
        ambiguity = alternative / max(best_score, 1e-9)
        accepted = (best_score >= c.min_score and ambiguity <= c.max_ambiguity
                    and overlap >= c.min_overlap)
        info.update(accepted=bool(accepted), score=best_score, fine_score=score,
                    ambiguity=ambiguity, overlap=overlap,
                    reason='accepted' if accepted else 'quality_gate')
        return (pose if accepted else None), info


# ---------------------------------------------------------------- graph

class Graph:
    def __init__(self, sessions, cfg):
        self.sessions, self.cfg = sessions, cfg
        self.nodes = []                       # (session index, keyframe index)
        self.offset = []
        for s, session in enumerate(sessions):
            self.offset.append(len(self.nodes))
            self.nodes += [(s, k) for k in range(len(session.keyframes))]
        self.poses = np.zeros((len(self.nodes), 3))
        self.placed = [False] * len(sessions)
        self.loop_edges = {}                  # (i, j) -> (measured, info)

    def node(self, s, k):
        return self.offset[s] + k

    def place(self, s, start):
        session = self.sessions[s]
        for k, f in enumerate(session.keyframes):
            self.poses[self.node(s, k)] = compose(start, session.raw[f])
        self.placed[s] = True

    def odometry_edges(self):
        c = self.cfg
        sigma = np.array([c.odom_sigma_m, c.odom_sigma_m, math.radians(c.odom_sigma_deg)])
        edges = []
        for s, session in enumerate(self.sessions):
            if not self.placed[s]:
                continue
            for k in range(1, len(session.keyframes)):
                a, b = session.keyframes[k - 1], session.keyframes[k]
                edges.append((self.node(s, k - 1), self.node(s, k),
                              relative(session.raw[a], session.raw[b]),
                              np.diag(1.0 / (sigma * math.sqrt(b - a)))))
        return edges

    def loop_info(self):
        c = self.cfg
        return np.diag(1.0 / np.array([c.loop_sigma_m, c.loop_sigma_m, math.radians(c.loop_sigma_deg)]))

    def optimize(self):
        active = [i for i, (s, _) in enumerate(self.nodes) if self.placed[s]]
        remap = {old: new for new, old in enumerate(active)}
        edges = [(remap[i], remap[j], m, w) for i, j, m, w in self.odometry_edges()]
        edges += [(remap[i], remap[j], m, self.loop_info())
                  for (i, j), (m, _) in self.loop_edges.items()]
        optimized = optimize_graph(self.poses[active], edges, self.cfg)
        self.poses[active] = optimized

    def residual_chi(self, i, j, measured):
        error = relative(self.poses[i], self.poses[j]) - measured
        error[2] = wrap_angle(error[2])
        return float(np.linalg.norm(self.loop_info() @ error))

    def frame_poses(self, s):
        session = self.sessions[s]
        out = np.zeros_like(session.raw)
        kf = 0
        for f in range(len(session.raw)):
            while kf + 1 < len(session.keyframes) and session.keyframes[kf + 1] <= f:
                kf += 1
            anchor = session.keyframes[kf]
            step = relative(session.raw[anchor], session.raw[f])
            step[:2] *= getattr(self, 'scale_of', {}).get(s, 1.0)
            out[f] = compose(self.poses[self.node(s, kf)], step)
        return out


def place_session(graph, matcher, s, log):
    """Global place recognition of session s against all placed sessions."""
    c = graph.cfg
    session = graph.sessions[s]
    hypotheses = []
    for kb in range(0, len(session.keyframes), c.place_stride):
        for t, other in enumerate(graph.sessions):
            if not graph.placed[t] or t == s:
                continue
            for ka in range(0, len(other.keyframes), c.place_stride):
                pose, info = matcher.match(session.submaps[kb], other.submaps[ka], (t, ka),
                                           np.zeros(3), 180, c.place_window_m)
                if pose is None:
                    continue
                world_b = compose(graph.poses[graph.node(t, ka)], pose)
                start = compose(world_b, inverse_pose(session.raw[session.keyframes[kb]]))
                hypotheses.append(dict(start=start, score=info['fine_score'], a=(t, ka), b=kb))
    best, support = None, []
    for h in hypotheses:
        inliers = [g for g in hypotheses
                   if np.hypot(*(g['start'][:2] - h['start'][:2])) <= c.consensus_m
                   and abs(math.degrees(wrap_angle(g['start'][2] - h['start'][2]))) <= c.consensus_deg]
        if len(inliers) > len(support) or (len(inliers) == len(support) and best is not None
                                           and h['score'] > best['score']):
            best, support = h, inliers
    log.append(dict(stage='place', session=session.name, hypotheses=len(hypotheses),
                    support=len(support)))
    if best is None or len(support) < c.min_consensus:
        return False
    graph.place(s, best['start'])
    for h in support:   # consistent place matches become loop edges
        t, ka = h['a']
        i, j = graph.node(t, ka), graph.node(s, h['b'])
        measured = relative(graph.poses[i], graph.poses[j])
        graph.loop_edges[(i, j)] = (measured, dict(kind='place', score=h['score']))
    return True


def search_loops(graph, matcher, tried, log):
    c = graph.cfg
    added = 0
    for j, (s, kb) in enumerate(graph.nodes):
        if not graph.placed[s]:
            continue
        sb = graph.sessions[s]
        candidates = []
        for i, (t, ka) in enumerate(graph.nodes[:j]):
            if not graph.placed[t] or (i, j) in tried:
                continue
            if t == s:
                sep = sb.path[sb.keyframes[kb]] - sb.path[sb.keyframes[ka]]
                if sep < c.loop_min_separation_m:
                    continue
                radius = c.loop_radius_m + c.loop_radius_per_m * sep
            else:
                radius = c.loop_radius_m + 0.5
            dist = float(np.hypot(*(graph.poses[i][:2] - graph.poses[j][:2])))
            if dist <= radius:
                candidates.append((dist, i, t, ka))
        for _, i, t, ka in sorted(candidates)[:c.loop_candidates_per_keyframe]:
            tried.add((i, j))
            initial = relative(graph.poses[i], graph.poses[j])
            pose, info = matcher.match(sb.submaps[kb], graph.sessions[t].submaps[ka], (t, ka),
                                       initial, 180, c.loop_window_m)
            if pose is not None:
                graph.loop_edges[(i, j)] = (pose, dict(kind='loop', **info))
                added += 1
    log.append(dict(stage='loops', added=added, total=len(graph.loop_edges)))
    return added


def prune_outliers(graph, log):
    removed = [key for key, (m, _) in graph.loop_edges.items()
               if graph.residual_chi(*key, m) > graph.cfg.outlier_chi]
    for key in removed:
        del graph.loop_edges[key]
    log.append(dict(stage='prune', removed=len(removed), remaining=len(graph.loop_edges)))
    return removed


def run_loop_slam(sessions, cfg):
    graph, matcher, log, tried = Graph(sessions, cfg), Matcher(cfg), [], set()
    order = sorted(range(len(sessions)), key=lambda s: sessions[s].timestamps[0])
    graph.place(order[0], np.zeros(3))
    for s in order[1:]:
        place_session(graph, matcher, s, log)
    graph.optimize()
    for _ in range(cfg.rounds):
        search_loops(graph, matcher, tried, log)
        graph.optimize()
        if prune_outliers(graph, log):
            graph.optimize()
    return graph, log


# ---------------------------------------------------------------- evaluation

def groundtruth_planar(dataset_root, sequence, timestamps):
    from .openloris_data import resolve_sequence_dir
    from .openloris_geometry import interpolate_pose, load_groundtruth
    gt = load_groundtruth(resolve_sequence_dir(dataset_root, sequence) / 'groundtruth.txt')
    out = []
    for t in timestamps:
        pose = interpolate_pose(gt, float(t), max_gap_sec=0.1)
        if pose is None:
            out.append([np.nan] * 3)
        else:
            out.append([pose[0, 3], pose[1, 3], math.atan2(pose[1, 0], pose[0, 0])])
    return np.asarray(out)


def fit_se2(source, target):
    """Rigid 2D transform (pose) mapping source xy to target xy (least squares)."""
    ms, mt = source.mean(0), target.mean(0)
    a, b = source - ms, target - mt
    yaw = math.atan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
    t = mt - rotate(ms[None], yaw)[0]
    return np.array([t[0], t[1], yaw])


def transform_traj(pose, traj):
    return np.array([compose(pose, p) for p in traj])


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a[:, :2] - b[:, :2]) ** 2, axis=1))))


def start_aligned(traj, gt):
    est = np.array([relative(traj[0], p) for p in traj])
    ref = np.array([relative(gt[0], p) for p in gt])
    yaw = wrap_angle(est[-1, 2] - ref[-1, 2])
    return dict(ate_m=rmse(est, ref), endpoint_m=float(np.hypot(*(est[-1, :2] - ref[-1, :2]))),
                endpoint_yaw_deg=abs(math.degrees(yaw)))


def load_sessions(inputs, cfg):
    sessions = []
    for predictions, manifest in inputs:
        with Path(predictions).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        with Path(manifest).open(encoding='utf-8') as stream:
            manifests = {pair_key(r): r for r in (json.loads(l) for l in stream if l.strip())}
        for sequence, chain in select_chains(rows):
            if len(chain) < 20:
                continue
            matching = [manifests[pair_key(r)] for r in chain]
            increments = [[float(r[k]) for k in ('pred_forward_m', 'pred_left_m', 'pred_yaw_rad')]
                          for r in chain]
            files = [matching[0]['bev_0']] + [r['bev_1'] for r in matching]
            times = [float(chain[0]['timestamp_0'])] + [float(r['timestamp_1']) for r in chain]
            name = f'{sequence}#{sum(s.sequence == sequence for s in sessions)}'
            sessions.append(Session(name, sequence, increments, files, times, cfg))
    return sessions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', nargs=2, action='append', required=True,
                        metavar=('PREDICTIONS', 'MANIFEST'))
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--set', nargs=2, action='append', default=[], metavar=('FIELD', 'VALUE'),
                        help='override a LoopConfig field')
    args = parser.parse_args(argv)
    cfg = LoopConfig()
    for field, value in args.set:
        setattr(cfg, field, type(getattr(cfg, field))(value))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sessions = load_sessions(args.input, cfg)
    graph, log = run_loop_slam(sessions, cfg)

    # ---- scoring (ground truth only from here on)
    gts = [groundtruth_planar(args.dataset_root, s.sequence, s.timestamps) for s in sessions]
    est = {i: graph.frame_poses(i) for i in range(len(sessions)) if graph.placed[i]}
    report = dict(config=asdict(cfg), inputs=args.input, log=log, sessions=[],
                  loop_edges=[dict(i=list(graph.nodes[i]), j=list(graph.nodes[j]), **info)
                              for (i, j), (_, info) in graph.loop_edges.items()])
    for i, s in enumerate(sessions):
        entry = dict(name=s.name, frames=len(s.raw), keyframes=len(s.keyframes),
                     path_m=float(s.path[-1]), placed=graph.placed[i],
                     frontend=start_aligned(s.raw, gts[i]))
        if graph.placed[i]:
            entry['slam'] = start_aligned(est[i], gts[i])
        report['sessions'].append(entry)
    placed = [i for i in est]
    all_est = np.concatenate([est[i] for i in placed])
    all_gt = np.concatenate([gts[i] for i in placed])
    world = fit_se2(all_est[:, :2], all_gt[:, :2])
    slam_world = {i: transform_traj(world, est[i]) for i in placed}
    report['joint'] = dict(
        sessions_placed=len(placed), sessions_total=len(sessions),
        slam_ate_single_alignment_m=rmse(np.concatenate([slam_world[i] for i in placed]), all_gt))
    # frontend baseline: each session placed at its true start pose (it cannot know it)
    front_world = {i: transform_traj(gts[i][0], np.array([relative(s.raw[0], p) for p in s.raw]))
                   for i, s in enumerate(sessions)}
    report['joint']['frontend_ate_true_start_m'] = rmse(
        np.concatenate([front_world[i] for i in range(len(sessions))]), np.concatenate(gts))
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
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float),
                                         encoding='utf-8')
    lines = ['# 回环 SLAM 后端（v4）评估', '',
             '后端不使用真值；真值仅用于事后评分。前端基线把每段放在真实起点（它本身无法知道）。', '',
             '| 段 | 路程 (m) | 关键帧 | 已定位 | 前端 ATE (m) | SLAM ATE (m) | 前端终点 (m) | SLAM 终点 (m) | 前端终点角 (°) | SLAM 终点角 (°) |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for e in report['sessions']:
        f, g = e['frontend'], e.get('slam', {})
        def fmt(d, k, p=3):
            return f'{d[k]:.{p}f}' if k in d else '—'
        lines.append(f"| {e['name']} | {e['path_m']:.1f} | {e['keyframes']} | {'是' if e['placed'] else '否'} | "
                     f"{fmt(f, 'ate_m')} | {fmt(g, 'ate_m')} | {fmt(f, 'endpoint_m')} | {fmt(g, 'endpoint_m')} | "
                     f"{fmt(f, 'endpoint_yaw_deg', 1)} | {fmt(g, 'endpoint_yaw_deg', 1)} |")
    j = report['joint']
    lines += ['', f"整体（所有段放进同一张地图，只做一次整体刚体对齐）：SLAM ATE {j['slam_ate_single_alignment_m']:.3f} m；"
              f"前端（每段用真实起点）ATE {j['frontend_ate_true_start_m']:.3f} m。已定位 {j['sessions_placed']}/{j['sessions_total']} 段。",
              f"回环边：{len(graph.loop_edges)} 条。"]
    (output / '报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(json.dumps(log, ensure_ascii=False))


if __name__ == '__main__':
    main()
