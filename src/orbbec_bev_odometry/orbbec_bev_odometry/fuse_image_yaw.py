"""Replace the BEV model's per-pair yaw by the image (epipolar) yaw.

Translation stays from the BEV model (metric scale).  The image yaw is used
when the essential-matrix fit has at least ``--min-inliers`` inliers and the
median feature flow is at least ``--min-flow-px`` (otherwise the epipolar
geometry is degenerate); else the pure-rotation fit (``--fallback rot``) or the
model yaw is used.  Monocular only; output schema equals the input CSV.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .train_keypoint_bev import _wrap, summarize_rows, write_outputs

NUMERIC = ('timestamp_0', 'timestamp_1', 'target_forward_m', 'target_left_m', 'target_yaw_rad', 'pred_forward_m',
           'pred_left_m', 'pred_yaw_rad', 'translation_error_m', 'yaw_error_deg', 'predicted_step_m',
           'target_step_m', 'confidence')


def fuse(rows, image, min_inliers, min_flow_px, fallback):
    by_key = {(r['sequence'], round(float(r['timestamp_0']), 6)): r for r in image}
    used = dict(ess=0, rot=0, model=0)
    for r in rows:
        im = by_key.get((r['sequence'], round(r['timestamp_0'], 6)))
        yaw, src = r['pred_yaw_rad'], 'model'
        if im is not None:
            flow = float(im['median_flow_px']) if im['median_flow_px'] else 0.0
            if im['ess_yaw_rad'] and int(im['ess_inliers']) >= min_inliers and flow >= min_flow_px:
                yaw, src = float(im['ess_yaw_rad']), 'ess'
            elif fallback == 'rot' and im['rot_yaw_rad'] and int(im['rot_inliers']) >= min_inliers:
                yaw, src = float(im['rot_yaw_rad']), 'rot'
        used[src] += 1
        r['pred_yaw_rad'] = yaw
        r['yaw_error_deg'] = abs(math.degrees(_wrap(yaw - r['target_yaw_rad'])))
    return used


def load(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        for k in NUMERIC:
            r[k] = float(r[k])
        r['false_step_over_10cm'] = int(r['false_step_over_10cm'])
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--image-yaw', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--min-inliers', type=int, default=30)
    parser.add_argument('--min-flow-px', type=float, default=1.0)
    parser.add_argument('--fallback', choices=('model', 'rot'), default='model')
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('output dir not empty')
    rows = load(args.predictions)
    with open(args.image_yaw, encoding='utf-8-sig', newline='') as stream:
        image = list(csv.DictReader(stream))
    used = fuse(rows, image, args.min_inliers, args.min_flow_px, args.fallback)
    summary = summarize_rows(rows)
    summary['image_yaw'] = dict(vars(args), used=used)
    write_outputs(rows, summary, out)
    print(json.dumps({k: v for k, v in summary.items() if k != 'trajectory'}, default=str))


if __name__ == '__main__':
    main()
