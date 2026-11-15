"""Per-pair pose from image matches + DA3 depth of the first frame (PnP).

Monocular only: SIFT matches between colour frames A and B; matched pixels of
A are lifted to 3D with A's DA3 depth; ``solvePnPRansac`` gives the camera
motion, converted to base_link (forward, left, yaw).  Only one depth map per
pair is used, so frame-to-frame DA3 scale inconsistency does not enter.
Targets are copied for evaluation only.  Output schema = prediction CSV.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from .appearance_slam import FrameSource
from .image_yaw import Features
from .multichannel_bev import OPENLORIS_COLOR_INTRINSICS, OPENLORIS_COLOR_SIZE, OPENLORIS_T_BASE_COLOR


def base_motion(rvec, tvec):
    """PnP gives X_B = R X_A + t (A coords -> B coords). Return base motion A->B."""
    r, _ = cv2.Rodrigues(rvec)
    t_c = np.eye(4)
    t_c[:3, :3], t_c[:3, 3] = r, tvec.ravel()
    t_c01 = np.linalg.inv(t_c)                       # pose of camera B in camera A
    tb = OPENLORIS_T_BASE_COLOR
    t_b = tb @ t_c01 @ np.linalg.inv(tb)
    return t_b[0, 3], t_b[1, 3], math.atan2(t_b[1, 0], t_b[0, 0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True, help='rows (pairs) to process; model columns kept as fallback')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--da3-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-depth', type=float, default=0.3)
    parser.add_argument('--max-depth', type=float, default=6.0)
    parser.add_argument('--reproj-px', type=float, default=2.0)
    parser.add_argument('--min-inliers', type=int, default=20)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error('output exists')
    fx, fy, cx, cy = OPENLORIS_COLOR_INTRINSICS
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    src = FrameSource(args.dataset_root, 'da3', args.da3_root)
    feats = Features(src, 2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    with open(args.predictions, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if args.limit:
        rows = rows[:args.limit]
    extra = ['pnp_ok', 'pnp_inliers', 'model_forward_m', 'model_left_m', 'model_yaw_rad']
    depth_cache = {}
    for n, r in enumerate(rows):
        r['model_forward_m'], r['model_left_m'], r['model_yaw_rad'] = r['pred_forward_m'], r['pred_left_m'], r['pred_yaw_rad']
        r['pnp_ok'], r['pnp_inliers'] = 0, 0
        p0, d0 = feats.get(r['sequence'], r['timestamp_0'])
        p1, d1 = feats.get(r['sequence'], r['timestamp_1'])
        if d0 is None or d1 is None or len(p0) < 8 or len(p1) < 8:
            continue
        knn = matcher.knnMatch(d0, d1, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 12:
            continue
        key = (r['sequence'], r['timestamp_0'])
        if key not in depth_cache:
            if len(depth_cache) > 8:
                depth_cache.pop(next(iter(depth_cache)))
            depth_cache[key] = src.depth(r['sequence'], r['timestamp_0'])
        depth = depth_cache[key]
        if depth is None:
            continue
        sx, sy = depth.shape[1] / OPENLORIS_COLOR_SIZE[0], depth.shape[0] / OPENLORIS_COLOR_SIZE[1]
        a = p0[[m.queryIdx for m in good]]
        b = p1[[m.trainIdx for m in good]]
        ui = np.clip((a[:, 0] * sx).astype(int), 0, depth.shape[1] - 1)
        vi = np.clip((a[:, 1] * sy).astype(int), 0, depth.shape[0] - 1)
        z = depth[vi, ui]
        ok = np.isfinite(z) & (z > args.min_depth) & (z < args.max_depth)
        if ok.sum() < 8:
            continue
        xyz = np.column_stack(((a[ok, 0] - cx) / fx * z[ok], (a[ok, 1] - cy) / fy * z[ok], z[ok]))
        found, rvec, tvec, inl = cv2.solvePnPRansac(xyz, b[ok], k, None, reprojectionError=args.reproj_px,
                                                    iterationsCount=300, confidence=0.999,
                                                    flags=cv2.SOLVEPNP_EPNP)
        if not found or inl is None or len(inl) < args.min_inliers:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(xyz[inl.ravel()], b[ok][inl.ravel()], k, None, rvec, tvec)
        f, l, y = base_motion(rvec, tvec)
        r['pred_forward_m'], r['pred_left_m'], r['pred_yaw_rad'] = repr(float(f)), repr(float(l)), repr(float(y))
        r['pnp_ok'], r['pnp_inliers'] = 1, int(len(inl))
        if n % 200 == 0:
            print(n, len(rows), flush=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=fields + extra)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
