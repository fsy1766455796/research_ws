"""固定权重，在验证集比较相关分布读出；不修改原模型和检查点。"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .learned_bev_dataset import BEVPairDataset
from .learned_bev_model import LearnedBEVOdometry, warp_current_to_previous
from .local_map_backend import compose, metrics, select_chains


def overlap_scores(model, previous, current, previous_mask, current_mask):
    """仅用于诊断：按候选有效重叠面积归一化，排除低重叠候选。"""
    matcher = model.matcher
    batch, channels, height, width = current.shape
    count = len(matcher.yaw_values)
    pose = current.new_zeros((batch, count, 3))
    pose[:, :, 2] = matcher.yaw_values
    expanded = (current * current_mask)[:, None].expand(-1, count, -1, -1, -1)
    rotated = warp_current_to_previous(expanded.reshape(-1, channels, height, width),
                                       pose.reshape(-1, 3), model.metric_bounds)
    masks = current_mask[:, None].expand(-1, count, -1, -1, -1)
    warped_mask = warp_current_to_previous(masks.reshape(-1, 1, height, width),
                                           pose.reshape(-1, 3), model.metric_bounds)
    expanded_previous = (previous * previous_mask)[:, None].expand(-1, count, -1, -1, -1)
    numerator = matcher._linear_correlation(expanded_previous.reshape(-1, channels, height, width), rotated)
    mask_previous = previous_mask[:, None].expand(-1, count, -1, -1, -1)
    overlap = matcher._linear_correlation(mask_previous.reshape(-1, 1, height, width), warped_mask)
    forward_resolution = (model.metric_bounds[1] - model.metric_bounds[0]) / height
    left_resolution = (model.metric_bounds[3] - model.metric_bounds[2]) / width
    dy = int(math.floor(matcher.max_translation_m / forward_resolution))
    dx = int(math.floor(matcher.max_translation_m / left_resolution))
    sl = (slice(None), slice(height - 1 - dy, height + dy), slice(width - 1 - dx, width + dx))
    numerator = numerator[sl].reshape(batch, count, 2 * dy + 1, 2 * dx + 1)
    overlap = overlap[sl].reshape_as(numerator)
    minimum = torch.minimum(previous_mask.sum((1, 2, 3)), current_mask.sum((1, 2, 3))) * .3
    minimum = minimum.clamp_min(10)
    score = numerator / overlap.clamp_min(1.)
    return score.masked_fill(overlap < minimum[:, None, None, None], -1e4)


def statistics(rows):
    target = np.degrees([r['target_yaw_rad'] for r in rows])
    predicted = np.degrees([r['pred_yaw_rad'] for r in rows])
    coarse = np.degrees([r['coarse_yaw_rad'] for r in rows])
    residual = np.degrees([r['residual_yaw_rad'] for r in rows])
    errors = (predicted - target + 180) % 360 - 180
    translation = [math.hypot(r['pred_forward_m'] - r['target_forward_m'],
                              r['pred_left_m'] - r['target_left_m']) for r in rows]
    bins = []
    for lo, hi in [(0, .3), (.3, 1), (1, 3), (3, 6), (6, 20), (20, 180)]:
        mask = (abs(target) >= lo) & (abs(target) < hi)
        if mask.any():
            bins.append(dict(range_deg=[lo, hi], count=int(mask.sum()),
                             yaw_mae_deg=float(np.mean(abs(errors[mask]))),
                             mean_abs_true_deg=float(np.mean(abs(target[mask]))),
                             mean_abs_pred_deg=float(np.mean(abs(predicted[mask])))))
    chains = []
    for sequence, chain in select_chains(rows):
        p, t = [np.zeros(3)], [np.zeros(3)]
        for row in chain:
            p.append(compose(p[-1], [row['pred_forward_m'], row['pred_left_m'], row['pred_yaw_rad']]))
            t.append(compose(t[-1], [row['target_forward_m'], row['target_left_m'], row['target_yaw_rad']]))
        chains.append(dict(sequence=sequence, pairs=len(chain), **metrics(np.asarray(p), np.asarray(t))))
    return dict(count=len(rows), yaw_mae_deg=float(np.mean(abs(errors))),
                yaw_rmse_deg=float(np.sqrt(np.mean(errors ** 2))),
                translation_mae_m=float(np.mean(translation)),
                translation_rmse_m=float(np.sqrt(np.mean(np.square(translation)))),
                coarse_abs_max_deg=float(max(abs(coarse))),
                coarse_yaw_mae_deg=float(np.mean(abs(coarse - target))),
                residual_saturated_fraction=float(np.mean(abs(residual) > .95)),
                mean_yaw_entropy=float(np.mean([r['yaw_entropy'] for r in rows])),
                mean_yaw_peak_probability=float(np.mean([r['yaw_peak_probability'] for r in rows])),
                bins=bins, chains=chains)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--overlap-diagnostic', action='store_true')
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LearnedBEVOdometry(**checkpoint['model_config']).to(device).eval()
    model.load_state_dict(checkpoint['model_state'])
    dataset = BEVPairDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    methods = {'original': 1., 'logits_x6': 6., 'logits_x24': 24., 'peak_coarse': None}
    if args.overlap_diagnostic:
        methods.update(overlap_peak=None, foreground_peak=None)
    rows = {name: [] for name in methods}
    with torch.inference_mode():
        offset = 0
        for batch in loader:
            previous, current = batch['previous'].to(device), batch['current'].to(device)
            pf, pc = model.encoder(previous)
            cf, cc = model.encoder(current)
            match = model.matcher(pc, cc)
            extra = {}
            if args.overlap_diagnostic:
                ones = torch.ones_like(pc[:, :1])
                extra['overlap_peak'] = overlap_scores(model, pc, cc, ones, ones)
                pm = F.adaptive_max_pool2d(previous[:, :1], pc.shape[-2:])
                cm = F.adaptive_max_pool2d(current[:, :1], cc.shape[-2:])
                extra['foreground_peak'] = overlap_scores(model, pc, cc, pm, cm)
            yg, fg, lg = torch.meshgrid(match['yaw_values'], match['forward_values'],
                                        match['left_values'], indexing='ij')
            coordinates = torch.stack((fg, lg, yg), dim=-1).reshape(-1, 3)
            for name, multiplier in methods.items():
                flat = extra.get(name, match['logits']).flatten(1)
                if multiplier is None:
                    probability = torch.zeros_like(flat).scatter_(1, flat.argmax(1)[:, None], 1.)
                else:
                    probability = torch.softmax(flat * multiplier, dim=1)
                coarse = probability @ coordinates
                aligned = warp_current_to_previous(cf, coarse, model.metric_bounds)
                residual, confidence, _ = model.refiner(pf, aligned)
                pose = (coarse + residual).cpu().numpy()
                cp, rs = coarse.cpu().numpy(), residual.cpu().numpy()
                marginal = probability.reshape_as(match['logits']).sum(dim=(2, 3))
                entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum(1) / math.log(marginal.shape[1])
                peaks = marginal.max(1).values.cpu().numpy()
                for k in range(len(pose)):
                    record = dataset.records[offset + k]
                    row = dict(sequence=record.sequence, timestamp_0=record.timestamp_0,
                               timestamp_1=record.timestamp_1,
                               target_forward_m=record.forward_m, target_left_m=record.left_m,
                               target_yaw_rad=record.yaw_rad,
                               pred_forward_m=float(pose[k, 0]), pred_left_m=float(pose[k, 1]),
                               pred_yaw_rad=float(pose[k, 2]), coarse_yaw_rad=float(cp[k, 2]),
                               residual_yaw_rad=float(rs[k, 2]), yaw_entropy=float(entropy[k]),
                               yaw_peak_probability=float(peaks[k]), confidence=float(confidence[k]))
                    rows[name].append(row)
            offset += len(previous)
            if offset % 200 == 0:
                print(f'已诊断 {offset}/{len(dataset)} 个验证帧对', flush=True)
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), manifest=str(Path(args.manifest).resolve()),
                  effective_logit_scale=float(model.matcher.logit_scale.detach().exp().clamp(max=100)),
                  refiner_limits=model.refiner.limits.cpu().tolist(), methods={})
    for name, values in rows.items():
        result['methods'][name] = statistics(values)
        with (output_dir / f'{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=values[0].keys())
            writer.writeheader()
            writer.writerows(values)
    (output_dir / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for name, value in result['methods'].items():
        print(name, json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
