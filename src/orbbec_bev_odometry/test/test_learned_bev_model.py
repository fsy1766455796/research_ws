import pytest


torch = pytest.importorskip("torch")

from orbbec_bev_odometry.learned_bev_model import SE2Correlation


def test_se2_correlation_translation_direction():
    previous = torch.zeros((1, 1, 21, 21))
    current = torch.zeros_like(previous)
    previous[0, 0, 10, 10] = 1.0
    current[0, 0, 8, 9] = 1.0
    matcher = SE2Correlation(
        (-1.05, 1.05, -1.05, 1.05),
        max_translation_m=0.4,
        max_yaw_deg=0.01,
        yaw_bins=3,
    )
    output = matcher(previous, current)
    flat_index = int(torch.argmax(output["logits"]))
    _, _, forward_count, left_count = output["logits"].shape
    remainder = flat_index % (forward_count * left_count)
    forward_index = remainder // left_count
    left_index = remainder % left_count

    assert torch.isclose(
        output["forward_values"][forward_index], torch.tensor(0.2)
    )
    assert torch.isclose(
        output["left_values"][left_index], torch.tensor(0.1)
    )
