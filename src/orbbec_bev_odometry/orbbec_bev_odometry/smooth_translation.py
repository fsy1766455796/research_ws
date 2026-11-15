"""Temporal smoothing of frontend translation increments (no ground truth).

Monocular DA3-BEV translation has large high-frequency noise (per-pair
forward error std ~5 cm at 0.2 s on validation, lag-1 autocorrelation 0.38).
Along every continuous chain the per-pair steps are expressed in a common
frame (using the predicted - e.g. gyro-fused - yaw), averaged over a centred
window of ``2k+1`` pairs, and converted back to the pair frame.  Yaw is left
untouched.  Pairs outside a chain of non-overlapping pairs are written back
unchanged.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .local_map_backend import pair_key, select_chains


def smooth_chain(chain, k):
    yaw = np.array([float(r['pred_yaw_rad']) for r in chain])
    heading = np.concatenate(([0.0], np.cumsum(yaw)))[:-1]       # heading at each pair start
    step = np.array([[float(r['pred_forward_m']), float(r['pred_left_m'])] for r in chain])
    c, s = np.cos(heading), np.sin(heading)
    world = np.column_stack((c * step[:, 0] - s * step[:, 1], s * step[:, 0] + c * step[:, 1]))
    out = np.empty_like(world)
    for i in range(len(world)):
        lo, hi = max(0, i - k), min(len(world), i + k + 1)
        out[i] = world[lo:hi].mean(0)
    for i, r in enumerate(chain):
        r['pred_forward_m'] = repr(float(c[i] * out[i, 0] + s[i] * out[i, 1]))
        r['pred_left_m'] = repr(float(-s[i] * out[i, 0] + c[i] * out[i, 1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', type=int, required=True, help='half window k (pairs)')
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error('output exists')
    with open(args.predictions, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for _, chain in select_chains(rows):      # rows are mutated in place
        smooth_chain(chain, args.window)
    with open(args.output, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
