"""Structural flow: coherent plane layouts, full connectivity (n<=12), vertical lift."""

from __future__ import annotations

import numpy as np

from apps.api.main import (
    STRUCTURAL_FLOW_PLANE_MAX_N,
    _build_geometry_edges,
    _diamond_plane_positions_four,
    _plane_ring_positions,
)


def test_full_connectivity_four_sensors() -> None:
    corr = np.full((4, 4), 0.4, dtype=float)
    np.fill_diagonal(corr, 1.0)
    names = ["a", "b", "c", "d"]
    edges = _build_geometry_edges(corr, feature_names=names, baseline_ref=None, limit=240)
    assert len(edges) == 6
    pairs = {(e["source"], e["target"]) for e in edges}
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = names[i], names[j]
            assert (a, b) in pairs or (b, a) in pairs


def test_diamond_lift_scales_with_stress_and_global_metrics() -> None:
    stress = np.array([0.05, 0.95, 0.08, 0.1], dtype=float)
    pos_lo = _diamond_plane_positions_four(stress, drift_global=0.0, inst_global=0.0)
    pos_hi = _diamond_plane_positions_four(stress, drift_global=0.8, inst_global=0.5)
    assert pos_lo[1, 1] > pos_lo[0, 1]
    assert pos_hi[1, 1] > pos_lo[1, 1]
    assert np.allclose(pos_lo[[0, 2, 3], 1], 0.0, atol=1e-6)


def test_full_connectivity_up_to_plane_max_nodes() -> None:
    n = STRUCTURAL_FLOW_PLANE_MAX_N
    corr = np.full((n, n), 0.35, dtype=float)
    np.fill_diagonal(corr, 1.0)
    names = [f"s{i}" for i in range(n)]
    edges = _build_geometry_edges(corr, feature_names=names, baseline_ref=None, limit=400)
    assert len(edges) == n * (n - 1) // 2


def test_plane_ring_positions() -> None:
    stress = np.array([0.1] * 9 + [0.55], dtype=float)
    pos = _plane_ring_positions(10, stress, drift_global=0.0, inst_global=0.0)
    assert pos.shape == (10, 3)
    assert pos[9, 1] > pos[0, 1]


def test_plane_dual_ring_positions() -> None:
    stress = np.array([0.1] * 15 + [0.6], dtype=float)
    pos = _plane_ring_positions(16, stress, drift_global=0.0, inst_global=0.0)
    assert pos.shape == (16, 3)
    # Inner ring smaller radius than outer
    ri = np.hypot(pos[0, 0], pos[0, 2])
    ro = np.hypot(pos[15, 0], pos[15, 2])
    assert ri < ro
    assert pos[15, 1] > pos[0, 1]


def test_threshold_still_limits_large_graphs() -> None:
    n = 26
    corr = np.eye(n, dtype=float) * 1.0
    corr[0, 1] = corr[1, 0] = 0.92
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) == (0, 1):
                continue
            v = 0.04 + (i + j) * 0.001
            corr[i, j] = corr[j, i] = v
    names = [f"s{i}" for i in range(n)]
    edges = _build_geometry_edges(corr, feature_names=names, baseline_ref=None, limit=240)
    assert len(edges) < n * (n - 1) // 2
