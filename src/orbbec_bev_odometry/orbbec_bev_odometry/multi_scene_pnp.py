"""v8 monocular frontend (SIFT + DA3-depth PnP + arc constraint) on any OpenLORIS sequence.

Per-sequence calibration is read from the sequence's own ``sensors.yaml``
(d400_color intrinsics, order fx, cx, fy, cy) and ``trans_matrix.yaml``
(base_link -> ... -> d400_color_optical_frame chain), because the scenes were
recorded with different mounts (e.g. market1 goes through ``laser``).
Pairs: every colour frame i with i+stride, both ends having interpolated
ground truth (targets used only for evaluation). PnP failures fall back to the
previous successful motion (constant velocity); optional physical-limit gate
(--max-step-m / --max-step-yaw-deg) treats implausible PnP results as failures. No training, no ground truth
in the estimate. Output: predictions.csv + summary.json (evaluate_learned_bev
trajectory metrics) per sequence plus a combined summary.
"""

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .evaluate_learned_bev import _wrap
from .openloris_data import _load_da3_depth, _resolve_da3_prediction, resolve_sequence_dir
from .openloris_geometry import interpolate_pose, load_groundtruth, load_timestamped_paths, relative_planar_pose
from .train_keypoint_bev import summarize_rows, write_outputs


def read_calibration(seq_dir):
    fs = cv2.FileStorage(str(seq_dir / 'sensors.yaml'), cv2.FILE_STORAGE_READ)
    node = fs.getNode('d400_color_optical_frame')
    fx, cx, fy, cy = node.getNode('intrinsics').mat().ravel()
    size = (int(node.getNode('width').real()), int(node.getNode('height').real()))
    fs.release()
    fs = cv2.FileStorage(str(seq_dir / 'trans_matrix.yaml'), cv2.FILE_STORAGE_READ)
    edges = {}
    tm = fs.getNode('trans_matrix')
    for i in range(tm.size()):
        n = tm.at(i)
        edges[n.getNode('child_frame').string()] = (n.getNode('parent_frame').string(), n.getNode('matrix').mat())
    fs.release()
    t, frame = np.eye(4), 'd400_color_optical_frame'
    while frame != 'base_link':
        parent, m = edges[frame]
        t, frame = m @ t, parent
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), size, t


def base_motion(rvec, tvec, t_bc):
    r, _ = cv2.Rodrigues(rvec)
    t_c = np.eye(4)
    t_c[:3, :3], t_c[:3, 3] = r, tvec.ravel()
    t_b = t_bc @ np.linalg.inv(t_c) @ np.linalg.inv(t_bc)
    return t_b[0, 3], t_b[1, 3], math.atan2(t_b[1, 0], t_b[0, 0])


def run_sequence(seq, args):
    d = resolve_sequence_dir(args.dataset_root, seq)
    k, size, t_bc = read_calibration(d)
    colors = load_timestamped_paths(d / 'color.txt')
    gt = load_groundtruth(d / 'groundtruth.txt')
    poses = [interpolate_pose(gt, c.timestamp, max_gap_sec=args.max_gt_gap) for c in colors]
    feats, depth_cache = {}, {}
    if args.matcher == 'sift':
        sift = cv2.SIFT_create(nfeatures=args.features)
        bf = cv2.BFMatcher(cv2.NORM_L2)
    else:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import numpy_image_to_torch
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        sp = SuperPoint(max_num_keypoints=args.features).eval().to(dev)
        lg = LightGlue(features='superpoint').eval().to(dev)

    def feat(i):
        if i not in feats:
            if len(feats) > 3 * args.stride:
                feats.pop(next(iter(feats)))
            if args.matcher == 'sift':
                img = cv2.imread(str(d / colors[i].path), cv2.IMREAD_GRAYSCALE)
                kp, des = sift.detectAndCompute(img, None)
                feats[i] = (np.array([p.pt for p in kp], np.float64).reshape(-1, 2), des)
            else:
                img = cv2.imread(str(d / colors[i].path), cv2.IMREAD_COLOR)[..., ::-1].copy()
                with torch.no_grad():
                    feats[i] = sp.extract(numpy_image_to_torch(img).to(dev))
        return feats[i]

    def match(i, j):
        """Matched pixel arrays (a in frame i, b in frame j) or None."""
        if args.matcher == 'sift':
            p0, d0 = feat(i)
            p1, d1 = feat(j)
            if d0 is None or d1 is None or len(p0) < 8 or len(p1) < 8:
                return None
            knn = bf.knnMatch(d0, d1, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
            if len(good) < 12:
                return None
            return p0[[m.queryIdx for m in good]], p1[[m.trainIdx for m in good]]
        f0, f1 = feat(i), feat(j)
        with torch.no_grad():
            m = lg({'image0': f0, 'image1': f1})['matches'][0].cpu().numpy()
        if len(m) < 12:
            return None
        k0 = f0['keypoints'][0].cpu().numpy().astype(np.float64)
        k1 = f1['keypoints'][0].cpu().numpy().astype(np.float64)
        return k0[m[:, 0]], k1[m[:, 1]]

    rows, last, n_ok, n_pairs, n_gated, t0 = [], (0.0, 0.0, 0.0), 0, 0, 0, time.time()
    for i in range(len(colors) - args.stride):
        j = i + args.stride
        motion, ok, ninl = None, 0, 0
        mt = match(i, j)
        if mt is not None:
            if i not in depth_cache:
                depth_cache.clear()
                f = _resolve_da3_prediction(args.da3_root, seq, colors[i].path, colors[i].timestamp)
                depth_cache[i] = None if f is None else _load_da3_depth(f)
            depth = depth_cache[i]
            if depth is not None:
                sx, sy = depth.shape[1] / size[0], depth.shape[0] / size[1]
                a, b = mt
                ui = np.clip((a[:, 0] * sx).astype(int), 0, depth.shape[1] - 1)
                vi = np.clip((a[:, 1] * sy).astype(int), 0, depth.shape[0] - 1)
                z = depth[vi, ui]
                sel = np.isfinite(z) & (z > args.min_depth) & (z < args.max_depth)
                if sel.sum() >= 8:
                    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
                    xyz = np.column_stack(((a[sel, 0] - cx) / fx * z[sel], (a[sel, 1] - cy) / fy * z[sel], z[sel]))
                    found, rvec, tvec, inl = cv2.solvePnPRansac(
                        xyz, b[sel], k, None, reprojectionError=args.reproj_px, iterationsCount=300,
                        confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
                    if found and inl is not None and len(inl) >= args.min_inliers:
                        rvec, tvec = cv2.solvePnPRefineLM(xyz[inl.ravel()], b[sel][inl.ravel()], k, None, rvec, tvec)
                        fwd, _, yaw = base_motion(rvec, tvec, t_bc)
                        if args.max_step_m > 0 and (abs(fwd) > args.max_step_m or abs(math.degrees(yaw)) > args.max_step_yaw_deg):
                            n_gated += 1
                        else:
                            motion, ok, ninl = (fwd, fwd * math.tan(yaw / 2), yaw), 1, len(inl)
        if motion is None:
            motion = last
        else:
            last = motion
        if poses[i] is None or poses[j] is None:
            continue
        n_pairs += 1
        n_ok += ok
        tgt = relative_planar_pose(poses[i], poses[j])
        pred = np.array(motion)
        ps, ts = float(np.linalg.norm(pred[:2])), float(np.linalg.norm(tgt[:2]))
        rows.append(dict(
            sequence=seq, timestamp_0=colors[i].timestamp, timestamp_1=colors[j].timestamp,
            target_forward_m=float(tgt[0]), target_left_m=float(tgt[1]), target_yaw_rad=float(tgt[2]),
            pred_forward_m=float(pred[0]), pred_left_m=float(pred[1]), pred_yaw_rad=float(pred[2]),
            translation_error_m=float(np.linalg.norm(pred[:2] - tgt[:2])),
            yaw_error_deg=abs(math.degrees(_wrap(float(pred[2] - tgt[2])))),
            predicted_step_m=ps, target_step_m=ts, false_step_over_10cm=int(ps > 0.10 and ts <= 0.10),
            confidence=float(ninl), pnp_ok=ok))
        if i % 500 == 0:
            print(seq, i, len(colors), f'{time.time() - t0:.0f}s', flush=True)
    return rows, dict(frames=len(colors), frames_with_gt=sum(p is not None for p in poses), pairs=n_pairs,
                      pnp_ok_ratio=n_ok / max(n_pairs, 1), gated=n_gated, matcher=args.matcher, intrinsics=k.tolist(),
                      t_base_color=t_bc.tolist(), seconds=time.time() - t0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--da3-root', required=True)
    parser.add_argument('--sequences', required=True, help='comma list')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--features', type=int, default=2000)
    parser.add_argument('--matcher', choices=('sift', 'superpoint_lightglue'), default='sift')
    parser.add_argument('--min-depth', type=float, default=0.3)
    parser.add_argument('--max-depth', type=float, default=6.0)
    parser.add_argument('--reproj-px', type=float, default=2.0)
    parser.add_argument('--min-inliers', type=int, default=20)
    parser.add_argument('--max-gt-gap', type=float, default=0.1)
    parser.add_argument('--max-step-m', type=float, default=0.0, help='physical limit per pair; 0 = off')
    parser.add_argument('--max-step-yaw-deg', type=float, default=30.0)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('output dir not empty')
    combined = {}
    for seq in args.sequences.split(','):
        rows, info = run_sequence(seq, args)
        if not rows:
            combined[seq] = dict(info, error='no pairs with ground truth')
            continue
        summary = summarize_rows(rows)
        summary['run'] = info
        write_outputs(rows, summary, out / seq)
        tr = summary['trajectory'][seq]
        combined[seq] = dict(info, ate_rmse_m=tr['ate_rmse_m'], target_distance_m=tr['target_distance_m'],
                             distance_ratio=tr['distance_ratio'], segment_count=tr['segment_count'],
                             time_coverage_ratio=tr['time_coverage_ratio'],
                             translation_mae_m=summary['translation_mae_m'], yaw_mae_deg=summary['yaw_mae_deg'])
        (out / 'combined.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
        print(json.dumps({seq: combined[seq]}, default=str), flush=True)
    (out / 'combined.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
