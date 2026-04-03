"""Smoke tests for optional platform timing helpers."""

from __future__ import annotations

import math

import numpy as np

from neraium_core.benchmarks.platform_hotpath import run_platform_benchmark
from neraium_core.graph import _undirected_graph_is_connected


def test_run_platform_benchmark_smoke() -> None:
    r = run_platform_benchmark(repeats=3, include_engine=False)
    assert math.isfinite(float(r["build_temporal_representation_s"]))
    assert math.isfinite(float(r["graph_metrics_s"]))
    assert "structural_engine_segment_s" not in r


def test_connectivity_vector_matches_matrix_power() -> None:
    rng = np.random.default_rng(7)
    for n in range(25, 36):
        for _ in range(40):
            a = rng.integers(0, 2, size=(n, n))
            a = np.maximum(a, a.T).astype(float)
            np.fill_diagonal(a, 0)
            reach = np.linalg.matrix_power(a + np.eye(n), max(n - 1, 1))
            legacy = float(np.all(reach > 0))
            fast = float(_undirected_graph_is_connected(a))
            assert legacy == fast
