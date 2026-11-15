"""Regression tests for segmented trajectory evaluation."""

import importlib
import json
import math
import subprocess
import sys

import pytest

from orbbec_bev_odometry.evaluate_learned_bev import _trajectory_metrics


def _row(start, end, pred=(0, 0, 0), target=(0, 0, 0), sequence="s"):
    return dict(
        sequence=sequence, timestamp_0=start, timestamp_1=end,
        pred_forward_m=pred[0], pred_left_m=pred[1],
        pred_yaw_rad=math.radians(pred[2]),
        target_forward_m=target[0], target_left_m=target[1],
        target_yaw_rad=math.radians(target[2]),
    )


def test_single_segment_preserves_all_legacy_fields():
    result = _trajectory_metrics([
        _row(1, 2, pred=(2, 0, 0), target=(1, 0, 0)),
        _row(0, 1, pred=(2, 0, 90), target=(1, 0, 90)),
    ])["s"]
    expected = dict(
        chain_pairs=2, ate_rmse_m=1.0,
        endpoint_translation_error_m=math.sqrt(2),
        endpoint_yaw_error_deg=0.0, predicted_distance_m=4.0,
        target_distance_m=2.0, distance_ratio=2.0,
        false_step_over_10cm_count=0,
    )
    for field, value in expected.items():
        assert result[field] == pytest.approx(value)
        assert result["segments"][0][field] == pytest.approx(value)
    assert result["segment_count"] == 1
    assert result["pair_coverage_ratio"] == 1
    assert result["time_coverage_ratio"] == 1


def test_gaps_restart_and_pool_all_segments_without_fake_endpoint():
    rows = [
        _row(0, 1, pred=(1, 0, 0)),
        _row(1, 2, pred=(1, 0, 90)),
        _row(5, 6, pred=(3, 0, -60)),
    ]
    result = _trajectory_metrics(list(reversed(rows)))["s"]
    assert result["input_pairs"] == result["chain_pairs"] == 3
    assert result["skipped_pairs"] == 0
    assert result["segment_count"] == 2
    assert result["pair_coverage_ratio"] == 1
    assert result["covered_duration_s"] == 3
    assert result["sequence_span_s"] == 6
    assert result["time_coverage_ratio"] == 0.5
    first, second = result["segments"]
    assert first["chain_pairs"] == 2
    assert second["chain_pairs"] == 1
    assert (first["timestamp_0"], first["timestamp_1"]) == (0, 2)
    assert (second["timestamp_0"], second["timestamp_1"]) == (5, 6)
    assert first["endpoint_translation_error_m"] == 2
    assert second["endpoint_translation_error_m"] == 3
    assert first["yaw_rmse_deg"] == pytest.approx(math.sqrt(90**2 / 3))
    assert second["yaw_rmse_deg"] == pytest.approx(math.sqrt(60**2 / 2))
    assert first["endpoint_cumulative_yaw_error_deg"] == pytest.approx(90)
    assert second["endpoint_cumulative_yaw_error_deg"] == pytest.approx(-60)
    assert result["ate_rmse_m"] == pytest.approx(math.sqrt(14 / 5))
    assert result["yaw_rmse_deg"] == pytest.approx(math.sqrt(11700 / 5))
    assert result["predicted_distance_m"] == 5
    assert result["target_distance_m"] == 0
    assert result["distance_ratio"] is None
    assert result["false_step_over_10cm_count"] == 3
    for field in ("endpoint_translation_error_m", "endpoint_yaw_error_deg",
                  "endpoint_cumulative_yaw_error_deg"):
        assert result[field] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_overlaps_duplicates_and_equal_starts_are_not_double_counted():
    rows = [_row(a, b, pred=(1, 0, 0)) for a, b in
            [(0, 4), (0, 2), (0, 2), (1, 3), (2, 4), (7, 9)]]
    result = _trajectory_metrics(rows)["s"]
    assert result == _trajectory_metrics(list(reversed(rows)))["s"]
    assert result["input_pairs"] == 6
    assert result["chain_pairs"] == result["skipped_pairs"] == 3
    assert result["pair_coverage_ratio"] == 0.5
    assert result["covered_duration_s"] == 6
    assert result["time_coverage_ratio"] == pytest.approx(6 / 9)
    assert [s["chain_pairs"] for s in result["segments"]] == [2, 1]
    assert result["predicted_distance_m"] == 3
    assert result["false_step_over_10cm_count"] == 3


@pytest.mark.parametrize("sign", [-1, 1])
def test_signed_cumulative_yaw_keeps_full_turns(sign):
    rows = [_row(i, i + 1, pred=(0, 0, sign * 110),
                 target=(0, 0, sign * 10)) for i in range(4)]
    result = _trajectory_metrics(rows)["s"]
    for metrics in (result, result["segments"][0]):
        assert metrics["endpoint_cumulative_yaw_error_deg"] == pytest.approx(
            sign * 400)
        assert metrics["endpoint_yaw_error_deg"] == pytest.approx(40)
        assert metrics["yaw_rmse_deg"] == pytest.approx(
            math.sqrt((100**2 + 160**2 + 60**2 + 40**2) / 5))


def test_matching_rotation_crossing_wrap_has_zero_yaw_error():
    result = _trajectory_metrics([
        _row(i, i + 1, pred=(0, 0, 120), target=(0, 0, 120))
        for i in range(4)
    ])["s"]
    assert result["yaw_rmse_deg"] == pytest.approx(0)
    assert result["endpoint_cumulative_yaw_error_deg"] == pytest.approx(0)


def test_timestamp_rounding_preserves_existing_continuity():
    result = _trajectory_metrics([
        _row(0, 1.00000001), _row(1.00000002, 2), _row(2.0000002, 3),
    ])["s"]
    assert [s["chain_pairs"] for s in result["segments"]] == [2, 1]


@pytest.mark.parametrize("start,end", [
    (1, 1), (2, 1), (1, 1.00000001),
    (float("nan"), 2), (0, float("inf")), (-float("inf"), 1),
])
def test_invalid_timestamps_fail_even_when_pair_would_be_skipped(start, end):
    with pytest.raises(ValueError, match="timestamps"):
        _trajectory_metrics([_row(0, 3), _row(start, end)])


def test_empty_input_and_independent_sequences():
    assert _trajectory_metrics([]) == {}
    result = _trajectory_metrics([
        _row(0, 1, pred=(2, 0, 90), sequence="a"),
        _row(0, 1, sequence="b"), _row(4, 5, sequence="a"),
    ])
    assert result["a"]["segment_count"] == 2
    assert result["b"]["segment_count"] == 1
    assert result["b"]["endpoint_translation_error_m"] == 0
    assert result["b"]["endpoint_yaw_error_deg"] == 0


@pytest.mark.parametrize("name", ["evaluate_icp_bev", "evaluate_ecc_bev"])
def test_existing_callers_share_helper_and_can_serialize_summary(name):
    module = importlib.import_module("orbbec_bev_odometry." + name)
    assert module._trajectory_metrics is _trajectory_metrics
    summary = {"trajectory": module._trajectory_metrics([
        _row(0, 1), _row(3, 4),
    ])}
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_evaluator_import_does_not_require_scipy_or_local_backend():
    code = """
import sys
sys.modules['scipy'] = None
sys.modules['orbbec_bev_odometry.local_map_backend'] = None
from orbbec_bev_odometry.evaluate_learned_bev import _trajectory_metrics
assert _trajectory_metrics([]) == {}
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
