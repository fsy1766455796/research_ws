import math

import torch
from torch.nn import functional as F

from orbbec_bev_odometry.learned_bev_model import LearnedBEVOdometry, SE2Correlation
from orbbec_bev_odometry.train_learned_bev import _load_pretrained, _compatible_resume_config


def test_cosine_scores_invariant_to_repeated_normalized_channels():
    torch.manual_seed(7)
    previous = F.normalize(torch.randn(1, 4, 9, 9), dim=1)
    current = F.normalize(torch.randn(1, 4, 9, 9), dim=1)
    matcher = SE2Correlation((-1, 1, -1, 1), max_translation_m=.3,
                             yaw_bins=3, correlation_mode='cosine')
    first = matcher(previous, current)['logits']
    repeated = matcher(previous.repeat(1, 2, 1, 1)/math.sqrt(2),
                       current.repeat(1, 2, 1, 1)/math.sqrt(2))['logits']
    torch.testing.assert_close(first, repeated)
    first.sum().backward()
    assert matcher.logit_scale.grad.abs() > 0


def test_pretrained_does_not_overwrite_new_limits_and_resets_saturated_yaw():
    old = LearnedBEVOdometry(9)
    new = LearnedBEVOdometry(9, yaw_residual_deg=8, correlation_mode='cosine')
    temperature = new.matcher.logit_scale.detach().clone()
    with torch.no_grad():
        old.refiner.pose_head[-1].bias[2] = 8
    reset = _load_pretrained(new, {'model_state': old.state_dict(), 'model_config': {}},
                             {'correlation_mode': 'cosine'})
    assert reset
    assert math.isclose(float(new.refiner.limits[2]), math.radians(8), rel_tol=1e-6)
    torch.testing.assert_close(new.matcher.logit_scale, temperature)
    assert torch.count_nonzero(new.refiner.pose_head[-1].weight[2]) == 0
    assert new.refiner.pose_head[-1].bias[2] == 0
    torch.testing.assert_close(new.refiner.pose_head[-1].weight[:2], old.refiner.pose_head[-1].weight[:2])


def test_legacy_transfer_preserves_every_tensor():
    old, new = LearnedBEVOdometry(9), LearnedBEVOdometry(9)
    assert not _load_pretrained(new, {'model_state': old.state_dict()}, {})
    for key, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], value)


def test_resume_accepts_legacy_defaults_but_rejects_new_architecture():
    assert _compatible_resume_config({'input_channels': 9},
                                    {'input_channels': 9, 'correlation_mode': 'legacy', 'yaw_residual_deg': None})
    assert not _compatible_resume_config({'input_channels': 9},
                                        {'input_channels': 9, 'yaw_residual_deg': 8})
