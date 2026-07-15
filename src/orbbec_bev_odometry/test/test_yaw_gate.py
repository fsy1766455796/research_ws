import math

import pytest

from orbbec_bev_odometry.yaw_filter import gate_yaw_motion


def run_stationary_steps(steps_deg, release_deg=1.0):
    pending = 0.0
    outputs = []
    for step_deg in steps_deg:
        output, pending = gate_yaw_motion(
            math.radians(step_deg), pending, 0.0, 20.0, 0.15, 0.10, release_deg
        )
        outputs.append(math.degrees(output))
    return outputs, math.degrees(pending)


def test_translation_accepts_small_real_yaw():
    output, pending = gate_yaw_motion(
        math.radians(0.2), math.radians(0.4), 50.0, 20.0, 0.15, 0.10, 1.0
    )
    assert math.degrees(output) == pytest.approx(0.2)
    assert pending == 0.0


def test_stationary_signed_jitter_cancels():
    outputs, pending_deg = run_stationary_steps([0.3, -0.3] * 10)
    assert outputs == pytest.approx([0.0] * 20)
    assert pending_deg == pytest.approx(0.0)


def test_stationary_slow_turn_releases_accumulated_yaw():
    outputs, pending_deg = run_stationary_steps([0.2] * 5)
    assert outputs[:4] == pytest.approx([0.0] * 4)
    assert outputs[4] == pytest.approx(1.0)
    assert pending_deg == pytest.approx(0.0)


def test_stationary_noise_below_input_deadband_is_ignored():
    outputs, pending_deg = run_stationary_steps([0.05] * 100)
    assert outputs == pytest.approx([0.0] * 100)
    assert pending_deg == pytest.approx(0.0)
