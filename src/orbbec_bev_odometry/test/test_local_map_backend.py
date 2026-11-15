import numpy as np
import pytest

from orbbec_bev_odometry.local_map_backend import (
    BackendConfig, compose, match_map, optimize_window, run_backend, select_chains,
)
from orbbec_bev_odometry.se2 import inverse, matrix, transform_points


def test_overlapping_pairs_and_gaps_are_not_concatenated():
    rows = [dict(sequence='s', timestamp_0=a, timestamp_1=b)
            for a, b in [(0, 2), (1, 3), (2, 4), (7, 9)]]
    chains = select_chains(rows)
    assert [[r['timestamp_0'] for r in c] for _, c in chains] == [[0, 2], [7]]


def test_rejected_observations_preserve_frontend():
    increments = np.array([[0.1, 0, 0.03]] * 12)
    raw, corrected, diagnostics = run_backend(
        increments, lambda i: np.empty((0, 2)), BackendConfig(window=3))
    np.testing.assert_allclose(corrected, raw)
    assert not any(d['accepted'] for d in diagnostics)


def test_matching_direction_and_degenerate_wall_rejection():
    wall1 = np.column_stack((np.full(250, 2.), np.linspace(-2, 2, 250)))
    target = np.random.default_rng(7).uniform([-2, -1], [3, 2], (400, 2))
    expected = np.array([0.08, -0.05, 0.035])
    source = transform_points(inverse(matrix(*expected)), target)
    pose, diagnostic = match_map(source, target, np.array([0.07, -0.04, 0.025]), BackendConfig())
    assert diagnostic['accepted']
    np.testing.assert_allclose(pose, expected, atol=0.002)
    _, diagnostic = match_map(wall1, wall1, np.zeros(3), BackendConfig())
    assert not diagnostic['accepted']


def test_graph_anchor_and_map_constraint():
    increments = np.array([[0.1, 0., 0.04]] * 4)
    poses = [np.zeros(3)]
    for step in increments:
        poses.append(compose(poses[-1], step))
    output, success = optimize_window(poses, increments, [(0, 4, np.array([0.4, 0, 0]))],
                                      0, BackendConfig())
    assert success
    np.testing.assert_array_equal(output[0], poses[0])
    assert abs(output[-1, 2]) < abs(poses[-1][2])
    unchanged, success = optimize_window(poses, increments, [], 0, BackendConfig())
    np.testing.assert_array_equal(unchanged, poses)


def test_current_frame_not_in_its_own_map(monkeypatch):
    import orbbec_bev_odometry.local_map_backend as backend
    seen = []

    def matcher(source, target, initial, cfg):
        seen.append(float(source[0, 0]))
        assert not np.any(np.isclose(target[:, 0], source[0, 0]))
        return initial, dict(accepted=False, reason='test')

    monkeypatch.setattr(backend, 'match_map', matcher)
    run_backend(np.zeros((5, 3)), lambda i: np.array([[float(i), 0.]]), BackendConfig(window=2))
    assert seen == [2, 3, 4, 5]


@pytest.mark.parametrize('strategy', ['fixed_history', 'legacy'])
def test_old_boundary_factor_participates_with_fixed_endpoint(strategy):
    increments = np.array([[1., 0., 0.]] * 4)
    poses = np.array([[float(i), 0., 0.] for i in range(5)])
    before = poses.copy()
    # Pose zero must be read from fixed history, not negative-indexed into the window.
    edges = [(0, 3, np.array([2.6, 0., 0.]))]
    output, success = optimize_window(
        poses, increments, edges, 2, BackendConfig(strategy=strategy))
    assert success
    np.testing.assert_array_equal(poses, before)
    np.testing.assert_array_equal(output[0], poses[2])
    if strategy == 'fixed_history':
        assert 2.6 < output[1, 0] < 2.99
        assert output[2, 0] < poses[4, 0]
    else:
        np.testing.assert_array_equal(output, poses[2:])


def test_multiwindow_retention_and_expiry(monkeypatch):
    import orbbec_bev_odometry.local_map_backend as backend

    original_optimize = backend.optimize_window
    snapshots = {}

    def optimizer(poses, increments, edges, start, cfg):
        snapshots[(cfg.strategy, len(poses) - 1)] = [(i, j) for i, j, _ in edges]
        return original_optimize(poses, increments, edges, start, cfg)

    def matcher(source, target, initial, cfg):
        measured = initial.copy()
        if source[0, 0] == 2:
            measured[0] = 1.6
        return measured, dict(accepted=True, reason='test')

    monkeypatch.setattr(backend, 'optimize_window', optimizer)
    monkeypatch.setattr(backend, 'match_map', matcher)
    results = {}
    for strategy in ('fixed_history', 'legacy'):
        _, results[strategy], diagnostics = run_backend(
            np.array([[1., 0., 0.]] * 7), lambda i: np.array([[float(i), 0.]]),
            BackendConfig(window=2, strategy=strategy))
        assert all(d['accepted'] for d in diagnostics[1:])

    assert (0, 2) in snapshots[('fixed_history', 3)]
    assert (0, 2) in snapshots[('fixed_history', 4)]
    assert (0, 2) not in snapshots[('fixed_history', 5)]
    assert (0, 2) not in snapshots[('legacy', 3)]
    for (strategy, index), edges in snapshots.items():
        start = max(0, index - 2)
        assert all((j if strategy == 'fixed_history' else i) >= start for i, j in edges)
    # Historical measurements keep influencing the solution after their anchor exits.
    assert results['fixed_history'][2, 0] < results['legacy'][2, 0] - 0.01
    assert results['fixed_history'][-1, 0] < results['legacy'][-1, 0] - 0.01


@pytest.mark.parametrize('strategy', ['fixed_history', 'legacy'])
def test_pure_yaw_correction_at_displaced_origin(strategy):
    source = np.random.default_rng(18).uniform([-1., -1.], [1., 1.], (400, 2))
    cfg = BackendConfig(strategy=strategy)
    for x in (0., 4.):
        initial = np.array([x, 0., 0.])
        expected = np.array([x, 0., np.deg2rad(2.)])
        target = transform_points(matrix(*expected), source)
        # At x=4 the left transform translates ~0.140 m despite zero robot motion.
        left = matrix(*expected) @ inverse(matrix(*initial))
        if x == 4.:
            assert np.linalg.norm(left[:2, 2]) > cfg.max_correction_m
        pose, diagnostic = match_map(source, target, initial, cfg)
        assert diagnostic['accepted'] == (strategy == 'fixed_history' or x == 0.)
        assert diagnostic['correction_m'] < 1e-6
        assert diagnostic['correction_deg'] == pytest.approx(2., abs=1e-5)
        np.testing.assert_allclose(pose, expected if diagnostic['accepted'] else initial, atol=1e-6)


def test_actual_robot_displacement_is_still_gated():
    # Sparse landmarks make the over-limit translation unambiguous for local ICP.
    source = np.random.default_rng(18).uniform([-10., -10.], [10., 10.], (400, 2))
    initial = np.array([4., 0., 0.])
    expected = np.array([4.14, 0., 0.])
    target = transform_points(matrix(*expected), source)
    cfg = BackendConfig()
    pose, diagnostic = match_map(source, target, initial, cfg)
    assert not diagnostic['accepted']
    assert diagnostic['correction_m'] > cfg.max_correction_m
    np.testing.assert_array_equal(pose, initial)


def test_expired_factors_do_not_change_window():
    increments = np.array([[1., 0., 0.]] * 4)
    poses = np.array([[float(i), 0., 0.] for i in range(5)])
    output, success = optimize_window(
        poses, increments, [(0, 1, np.zeros(3))], 2, BackendConfig())
    assert success
    np.testing.assert_array_equal(output, poses[2:])


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError, match='strategy'):
        BackendConfig(strategy='unknown')
