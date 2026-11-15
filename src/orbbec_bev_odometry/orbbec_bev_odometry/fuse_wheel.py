"""Use the wheel odometer (OpenLORIS ``odom.txt``, 20 Hz) for translation scale.

``--mode scale``  : keep the visual translation direction of every pair,
                    set its length to the wheel-measured travelled distance.
``--mode replace``: translation entirely from the wheel odometer, expressed in
                    the pair start frame (reference only: no visual translation).
Yaw is left as in the input (e.g. already gyro-fused).  No ground truth used.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .openloris_data import resolve_sequence_dir


class Wheel:
    def __init__(self, sequence_dir):
        o = np.loadtxt(Path(sequence_dir) / 'odom.txt', comments='#')
        o = o[np.argsort(o[:, 0])]
        self.t, self.x, self.y = o[:, 0], o[:, 1], o[:, 2]
        self.yaw = np.unwrap(2 * np.arctan2(o[:, 6], o[:, 7]))

    def relative(self, t0, t1):
        x0, x1 = np.interp([t0, t1], self.t, self.x)
        y0, y1 = np.interp([t0, t1], self.t, self.y)
        a0 = float(np.interp(t0, self.t, self.yaw))
        dx, dy = x1 - x0, y1 - y0
        c, s = math.cos(a0), math.sin(a0)
        return c * dx + s * dy, -s * dx + c * dy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('scale', 'replace'), default='scale')
    parser.add_argument('--min-visual-m', type=float, default=0.005,
                        help='below this visual length the wheel direction is used')
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error('output exists')
    with open(args.predictions, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    wheels = {}
    for r in rows:
        seq = r['sequence']
        if seq not in wheels:
            wheels[seq] = Wheel(resolve_sequence_dir(args.dataset_root, seq))
        wf, wl = wheels[seq].relative(float(r['timestamp_0']), float(r['timestamp_1']))
        dist = math.hypot(wf, wl)
        vf, vl = float(r['pred_forward_m']), float(r['pred_left_m'])
        vlen = math.hypot(vf, vl)
        if args.mode == 'replace' or vlen < args.min_visual_m:
            nf, nl = wf, wl
        else:
            nf, nl = vf / vlen * dist, vl / vlen * dist
        r['pred_forward_m'], r['pred_left_m'] = repr(float(nf)), repr(float(nl))
    with open(args.output, 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'mode': args.mode, 'rows': len(rows)}))


if __name__ == '__main__':
    main()
