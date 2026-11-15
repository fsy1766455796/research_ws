"""Read-only checkpoint diagnostics; never train or replace checkpoints."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .learned_bev_dataset import BEVPairDataset
from .learned_bev_model import LearnedBEVOdometry, SE2Correlation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = LearnedBEVOdometry(**checkpoint['model_config']).eval()
    model.load_state_dict(checkpoint['model_state'])
    dataset = BEVPairDataset(args.manifest)
    index = next(i for i, r in enumerate(dataset.records)
                 if r.sequence == 'office1-5' and abs(r.timestamp_0 - 1560116839.46437) < 1e-6)
    sample = dataset[index]
    output = model(sample['previous'][None], sample['current'][None])
    target = sample['target']
    log_scale = model.matcher.logit_scale
    effective = log_scale.exp().clamp(max=100)
    scale_grad = torch.autograd.grad(effective, log_scale, retain_graph=True)[0]
    logits = output['logits'].flatten()
    residual_ratio = output['residual'][0, 2] / model.refiner.limits[2]
    example = dict(target_deg=math.degrees(float(target[2])),
                   coarse_deg=math.degrees(float(output['coarse_pose'][0, 2].detach())),
                   residual_deg=math.degrees(float(output['residual'][0, 2].detach())),
                   prediction_deg=math.degrees(float(output['pose'][0, 2].detach())),
                   residual_tanh_derivative=float((1-residual_ratio**2).detach()),
                   logit_range=float((logits.max()-logits.min()).detach()))
    with Path(args.diagnostics).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    true = np.degrees([float(r['target_yaw_rad']) for r in rows])
    coarse = np.degrees([float(r['coarse_yaw_rad']) for r in rows])
    residual = np.degrees([float(r['residual_yaw_rad']) for r in rows])
    limit = math.degrees(float(model.refiner.limits[2]))
    turn = abs(true) > 3
    lower_bound = np.maximum(abs(true-coarse)-limit, 0)
    # An ideal feature field with a known rigid transform isolates the readout
    # from learned feature quality and from depth estimation errors.
    forward, left = torch.meshgrid(torch.linspace(-.9, 5.9, 35),
                                  torch.linspace(-3.4, 3.4, 35), indexing='ij')
    frequencies = torch.from_numpy(np.random.default_rng(7).uniform(-5, 5, (24, 2))).float()
    def feature(x, y):
        return F.normalize(torch.sin(frequencies[:, 0, None, None]*x +
                                     frequencies[:, 1, None, None]*y +
                                     torch.arange(24)[:, None, None])[None], dim=1)
    matcher = SE2Correlation(model.metric_bounds)
    matcher.load_state_dict(model.matcher.state_dict())
    analytic = []
    with torch.no_grad():
        for degree in [-6, 6]:
            angle = math.radians(degree)
            previous = feature(forward, left)
            current = feature(math.cos(angle)*forward-math.sin(angle)*left+.2,
                              math.sin(angle)*forward+math.cos(angle)*left-.2)
            result = matcher(previous, current)
            peak = np.unravel_index(int(result['logits'].argmax()), result['logits'].shape)
            sharper = torch.softmax(result['logits'].flatten(1)*24, dim=1).reshape_as(result['logits'])
            sharper_yaw = (sharper * result['yaw_values'][None, :, None, None]).sum()
            analytic.append(dict(true_yaw_deg=degree,
                                 peak_yaw_deg=math.degrees(float(result['yaw_values'][peak[1]])),
                                 expectation_yaw_deg=math.degrees(float(result['pose'][0, 2])),
                                 expectation_without_channel_divisor_deg=math.degrees(float(sharper_yaw))))
    summary = dict(checkpoint=args.checkpoint, manifest=args.manifest,
                   checkpoint_epoch=checkpoint.get('epoch'),
                   raw_temperature_scale=float(log_scale.exp().detach()),
                   effective_temperature_scale=float(effective.detach()),
                   temperature_parameter_gradient=float(scale_grad),
                   residual_limit_deg=limit, example=example,
                   validation=dict(count=len(rows), turn_count=int(turn.sum()),
                     unreachable_fraction=float(np.mean(abs(true-coarse)>limit)),
                     turn_unreachable_fraction=float(np.mean(abs(true[turn]-coarse[turn])>limit)),
                     turn_residual_saturation_fraction=float(np.mean(abs(residual[turn])>.95*limit)),
                     fixed_coarse_best_possible_yaw_mae_deg=float(lower_bound.mean()),
                     turn_fixed_coarse_best_possible_yaw_mae_deg=float(lower_bound[turn].mean())),
                   analytic_features=analytic)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
