"""Per-pair yaw from the colour images alone (monocular, scale-free).

For every pair of a prediction CSV the two colour frames are matched (SIFT +
ratio test) and the relative rotation is estimated in two ways:

* ``ess``: essential matrix (RANSAC) + ``recoverPose`` (general motion);
* ``rot``: pure-rotation fit on unit bearing vectors (2-point RANSAC + Kabsch),
  valid when translation is small relative to scene depth.

The camera rotation is expressed in base_link (``OPENLORIS_T_BASE_COLOR``) and
reduced to yaw.  No depth, no ground truth (targets are only copied for
evaluation).  Output: CSV with one row per input pair.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from .appearance_slam import FrameSource
from .multichannel_bev import OPENLORIS_COLOR_INTRINSICS, OPENLORIS_T_BASE_COLOR


class Features:
    def __init__(self, source, n_features):
        self.source = source
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.cache = {}

    def get(self, sequence, timestamp):
        key = (sequence, round(float(timestamp), 6))
        if key not in self.cache:
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
            img = cv2.imread(str(self.source.color_path(sequence, timestamp)), cv2.IMREAD_GRAYSCALE)
            kp, des = self.sift.detectAndCompute(img, None)
            pts = np.array([k.pt for k in kp], np.float64).reshape(-1, 2)
            self.cache[key] = (pts, des)
        return self.cache[key]


def kabsch(a, b):
    """R minimising |b - R a| for row vectors a, b [N,3]."""
    h = a.T @ b
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ np.diag([1, 1, d]) @ u.T


def rotation_ransac(b0, b1, thresh_rad, iters, rng):
    best, best_in = None, None
    n = len(b0)
    cos_t = math.cos(thresh_rad)
    for _ in range(iters):
        i = rng.choice(n, 2, replace=False)
        r = kabsch(b0[i], b1[i])
        inl = np.einsum('ij,ij->i', (r @ b0.T).T, b1) > cos_t
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in is None or best_in.sum() < 3:
        return None, 0
    return kabsch(b0[best_in], b1[best_in]), int(best_in.sum())


def base_yaw(r_c1_from_c0):
    """r maps cam0 coordinates to cam1 coordinates -> yaw of base1 in base0."""
    r_bc = OPENLORIS_T_BASE_COLOR[:3, :3]
    r_c01 = r_c1_from_c0.T                       # orientation of cam1 in cam0
    r_b = r_bc @ r_c01 @ r_bc.T
    return math.atan2(r_b[1, 0], r_b[0, 0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--features', type=int, default=2000)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error('output exists')
    fx, fy, cx, cy = OPENLORIS_COLOR_INTRINSICS
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    kinv = np.linalg.inv(k)
    feats = Features(FrameSource(args.dataset_root, 'da3'), args.features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(0)
    with open(args.predictions, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if args.limit:
        rows = rows[:args.limit]
    out = []
    for n, r in enumerate(rows):
        p0, d0 = feats.get(r['sequence'], r['timestamp_0'])
        p1, d1 = feats.get(r['sequence'], r['timestamp_1'])
        rec = dict(sequence=r['sequence'], timestamp_0=r['timestamp_0'], timestamp_1=r['timestamp_1'],
                   target_yaw_rad=r['target_yaw_rad'], matches=0, ess_yaw_rad='', ess_inliers=0,
                   rot_yaw_rad='', rot_inliers=0, median_flow_px='')
        if d0 is not None and d1 is not None and len(p0) > 8 and len(p1) > 8:
            knn = matcher.knnMatch(d0, d1, k=2)
            good = [m[0] for m in knn if len(m) == 2 and m[0].distance < args.ratio * m[1].distance]
            rec['matches'] = len(good)
            if len(good) >= 12:
                a = p0[[m.queryIdx for m in good]]
                b = p1[[m.trainIdx for m in good]]
                rec['median_flow_px'] = float(np.median(np.linalg.norm(a - b, axis=1)))
                e, mask = cv2.findEssentialMat(a, b, k, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if e is not None and e.shape == (3, 3):
                    inl, rr, _, _ = cv2.recoverPose(e, a, b, k, mask=mask)
                    rec['ess_yaw_rad'], rec['ess_inliers'] = base_yaw(rr), int(inl)
                ha = np.column_stack((a, np.ones(len(a)))) @ kinv.T
                hb = np.column_stack((b, np.ones(len(b)))) @ kinv.T
                ha /= np.linalg.norm(ha, axis=1, keepdims=True)
                hb /= np.linalg.norm(hb, axis=1, keepdims=True)
                rr, ni = rotation_ransac(ha, hb, 2.0 / fx, 200, rng)
                if rr is not None:
                    rec['rot_yaw_rad'], rec['rot_inliers'] = base_yaw(rr), ni
        out.append(rec)
        if n % 200 == 0:
            print(n, len(rows), flush=True)
    with open(args.output, 'w', encoding='utf-8', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)


if __name__ == '__main__':
    main()
