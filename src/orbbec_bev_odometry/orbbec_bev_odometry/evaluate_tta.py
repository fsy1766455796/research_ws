"""Test-time augmentation (TTA) for ``KeypointBEVOdometry`` (monocular only).

Each pair (A, B) is predicted up to four times and the SE(2) estimates are
averaged (translation mean, circular yaw mean):

* ``fwd``  : model(A, B)
* ``rev``  : inverse of model(B, A)          (training used reversed pairs)
* ``flip`` : model(mirror A, mirror B) mirrored back (left axis flipped,
             BEV left range is symmetric -3.5..3.5 m): (x, -y, -yaw)
* ``fliprev``: both

No extra sensors, no ground truth.  The output schema equals
``train_keypoint_bev evaluate`` so every downstream tool works.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .learned_bev_dataset import BEVPairDataset
from .train_keypoint_bev import _wrap, build_model, summarize_rows, write_outputs


def se2_inverse(p):
    x, y, th = p[:, 0], p[:, 1], p[:, 2]
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack((-(c * x + s * y), -(-s * x + c * y), -th), -1)


def mirror(p):
    return torch.stack((p[:, 0], -p[:, 1], -p[:, 2]), -1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--modes', default='fwd,rev', help='comma list of fwd,rev,flip,fliprev')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('output dir not empty')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ck['model_config'], device)
    model.load_state_dict(ck['model_state'])
    model.eval()
    modes = args.modes.split(',')
    loader = DataLoader(BEVPairDataset(args.manifest), batch_size=args.batch_size, num_workers=2)
    rows = []
    with torch.no_grad():
        for batch in loader:
            a = batch['previous'].to(device)
            b = batch['current'].to(device)
            preds, confs = [], []

            def run(x, y):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    o = model(x, y)
                return o['pose'].float(), o['confidence'].float()

            for m in modes:
                if m == 'fwd':
                    p, c = run(a, b)
                elif m == 'rev':
                    p, c = run(b, a)
                    p = se2_inverse(p)
                elif m == 'flip':
                    p, c = run(a.flip(-1), b.flip(-1))
                    p = mirror(p)
                elif m == 'fliprev':
                    p, c = run(b.flip(-1), a.flip(-1))
                    p = se2_inverse(mirror(p))
                else:
                    raise ValueError(m)
                preds.append(p)
                confs.append(c)
            P = torch.stack(preds)                      # [M, B, 3]
            xy = P[..., :2].mean(0)
            yaw = torch.atan2(torch.sin(P[..., 2]).mean(0), torch.cos(P[..., 2]).mean(0))
            pred = torch.cat((xy, yaw[:, None]), -1).cpu().numpy()
            conf = torch.stack(confs).mean(0).cpu().numpy()
            target = batch['target'].numpy()
            for i in range(len(pred)):
                ps, ts = float(np.linalg.norm(pred[i, :2])), float(np.linalg.norm(target[i, :2]))
                rows.append({
                    'sequence': batch['sequence'][i],
                    'timestamp_0': float(batch['timestamp_0'][i]),
                    'timestamp_1': float(batch['timestamp_1'][i]),
                    'target_forward_m': float(target[i, 0]), 'target_left_m': float(target[i, 1]),
                    'target_yaw_rad': float(target[i, 2]),
                    'pred_forward_m': float(pred[i, 0]), 'pred_left_m': float(pred[i, 1]),
                    'pred_yaw_rad': float(pred[i, 2]),
                    'translation_error_m': float(np.linalg.norm(pred[i, :2] - target[i, :2])),
                    'yaw_error_deg': abs(math.degrees(_wrap(float(pred[i, 2] - target[i, 2])))),
                    'predicted_step_m': ps, 'target_step_m': ts,
                    'false_step_over_10cm': int(ps > 0.10 and ts <= 0.10),
                    'confidence': float(conf[i]),
                })
    summary = summarize_rows(rows)
    summary['tta_modes'] = modes
    write_outputs(rows, summary, out)
    print(json.dumps({k: v for k, v in summary.items() if k != 'trajectory'}))


if __name__ == '__main__':
    main()
