"""Fine-tune ``KeypointBEVOdometry`` with a multi-step (sequence) loss.

Why: pair-only training leaves small systematic biases (monocular DA3-BEV
under-turns in curves and under-estimates translation); they are invisible in
one 0.2 s pair but dominate the integrated trajectory.  Here each sample is a
window of ``N`` consecutive non-overlapping pairs.  The model predicts every
pair, the predictions are composed (differentiably) into poses relative to
the window start, and the loss covers

* the usual per-pair keypoint loss (``keypoint_bev_loss``), and
* ``sequence_weight`` x accumulated-pose error at every step k = 1..N
  (translation smooth-L1 + ``yaw_weight`` x yaw smooth-L1).

Validation, checkpoint selection and outputs are identical to
``train_keypoint_bev`` (validation trajectory ATE + 0.01 x trajectory yaw).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .keypoint_bev_loss import keypoint_bev_loss
from .learned_bev_dataset import BEVPairDataset
from .learned_bev_loss import wrap_angle
from .openloris_data import load_pair_manifest
from .train_keypoint_bev import build_model, predict_rows, selection_score, summarize_rows, write_outputs


class WindowDataset(Dataset):
    def __init__(self, manifest, steps):
        records = load_pair_manifest(manifest)
        by_start = {(r.sequence, round(r.timestamp_0, 6)): r for r in records}
        self.windows = []
        for r in records:
            chain, cur = [r], r
            while len(chain) < steps:
                nxt = by_start.get((cur.sequence, round(cur.timestamp_1, 6)))
                if nxt is None:
                    break
                chain.append(nxt)
                cur = nxt
            if len(chain) == steps:
                self.windows.append(chain)
        if not self.windows:
            raise ValueError('no windows')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        chain = self.windows[index]
        files = [chain[0].bev_0] + [r.bev_1 for r in chain]
        bevs = np.stack([BEVPairDataset._load_bev(f) for f in files])
        targets = np.stack([r.target for r in chain]).astype(np.float32)
        return {'bevs': torch.from_numpy(bevs), 'targets': torch.from_numpy(targets)}


def compose_steps(steps):
    """steps [B, N, 3] -> accumulated poses [B, N, 3] relative to the window start."""
    x = torch.zeros_like(steps[:, 0, 0])
    y, th = torch.zeros_like(x), torch.zeros_like(x)
    out = []
    for k in range(steps.shape[1]):
        dx, dy, dth = steps[:, k, 0], steps[:, k, 1], steps[:, k, 2]
        c, s = torch.cos(th), torch.sin(th)
        x, y, th = x + c * dx - s * dy, y + s * dx + c * dy, th + dth
        out.append(torch.stack((x, y, th), -1))
    return torch.stack(out, 1)


def sequence_loss(pred_steps, true_steps, yaw_weight):
    p, g = compose_steps(pred_steps), compose_steps(true_steps)
    translation = F.smooth_l1_loss(p[..., :2], g[..., :2], beta=0.05)
    yaw_error = wrap_angle(p[..., 2] - g[..., 2])
    yaw = F.smooth_l1_loss(yaw_error, torch.zeros_like(yaw_error), beta=math.radians(1.0))
    return translation, yaw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-manifest', required=True, help='stride-consistent pairs (e.g. stride 6)')
    parser.add_argument('--validation-manifest', required=True)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--windows-per-epoch', type=int, default=1500)
    parser.add_argument('--batch-windows', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--sequence-weight', type=float, default=1.0)
    parser.add_argument('--yaw-weight', type=float, default=4.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
    model_config = checkpoint['model_config']
    model = build_model(model_config, device)
    model.load_state_dict(checkpoint['model_state'])
    loss_config = dict(yaw_weight=args.yaw_weight, correspondence_weight=0.2, first_pass_weight=0.5)
    windows = WindowDataset(args.train_manifest, args.steps)
    validation = DataLoader(BEVPairDataset(args.validation_manifest), batch_size=4, num_workers=2,
                            pin_memory=device.type == 'cuda')
    steps_per_epoch = args.windows_per_epoch // args.batch_windows
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                    total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(device=str(device), windows=len(windows), model_config=model_config, loss_config=loss_config,
                args=vars(args))
    (output / 'config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)
    best = float('inf')
    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        sampler = torch.utils.data.RandomSampler(windows, replacement=False,
                                                 num_samples=min(args.windows_per_epoch, len(windows)))
        loader = DataLoader(windows, batch_size=args.batch_windows, sampler=sampler, num_workers=args.workers,
                            pin_memory=device.type == 'cuda', drop_last=True)
        sums, count = {}, 0
        for batch in loader:
            bevs = batch['bevs'].to(device, non_blocking=True)          # [B, N+1, C, H, W]
            targets = batch['targets'].to(device, non_blocking=True)    # [B, N, 3]
            b, n1 = bevs.shape[:2]
            n = n1 - 1
            previous = bevs[:, :-1].reshape(b * n, *bevs.shape[2:])
            current = bevs[:, 1:].reshape(b * n, *bevs.shape[2:])
            flat_targets = targets.reshape(b * n, 3)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                out = model(previous, current)
            parts = keypoint_bev_loss(out, flat_targets, **loss_config)
            seq_t, seq_y = sequence_loss(out['pose'].float().reshape(b, n, 3), targets.float(), args.yaw_weight)
            loss = parts['loss'] + args.sequence_weight * (seq_t + args.yaw_weight * seq_y)
            if not torch.isfinite(loss):
                scheduler.step()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            count += 1
            for key, value in (('loss', loss), ('pair_loss', parts['loss']), ('seq_translation', seq_t),
                               ('seq_yaw', seq_y)):
                sums[key] = sums.get(key, 0.0) + float(value.detach())
        rows = predict_rows(model, validation, device)
        summary = summarize_rows(rows)
        score = selection_score(summary)
        record = dict(epoch=epoch, seconds=time.time() - started, train={k: v / max(count, 1) for k, v in sums.items()},
                      validation={k: v for k, v in summary.items() if k != 'trajectory'}, validation_score=score)
        print(json.dumps(record), flush=True)
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        state = dict(epoch=epoch, model_type='keypoint_bev', model_config=model_config, model_state=model.state_dict(),
                     loss_config=loss_config, validation_score=score, sequence_args=vars(args))
        torch.save(state, output / 'last.pt')
        if score < best:
            best = score
            torch.save(state, output / 'best.pt')
            write_outputs(rows, summary, output / 'best_validation')


if __name__ == '__main__':
    main()
