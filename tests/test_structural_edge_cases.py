from __future__ import annotations

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.directional import directional_metrics
from neraium_core.geometry import correlation_matrix
from neraium_core.graph import graph_metrics
from neraium_core.spectral import spectral_gap, spectral_radius
from neraium_core.subsystems import discover_subsystems, subsystem_spectral_measures


def _frame(timestamp: str, sensor_values: dict[str, float | None]) -> dict:
    return {
        "timestamp": timestamp,
        "site_id": "site-a",
        "asset_id": "asset-a",
        "sensor_values": sensor_values,
    }


def test_relational_primitives_guard_degenerate_shapes() -> None:
    empty_obs = np.empty((4, 0), dtype=float)
    single_obs = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)

    corr_empty = correlation_matrix(empty_obs)
    corr_single = correlation_matrix(single_obs)

    assert corr_empty.shape == (0, 0)
    assert corr_single.shape == (1, 1)

    assert spectral_radius(corr_empty) == 0.0
    assert spectral_radius(corr_single) == 0.0
    assert spectral_gap(corr_empty) == 0.0
    assert spectral_gap(corr_single) == 0.0

    assert directional_metrics(corr_empty)["causal_divergence"] == 0.0
    assert directional_metrics(corr_single)["causal_divergence"] == 0.0

    assert discover_subsystems(corr_empty) == []
    assert discover_subsystems(corr_single) == []

    assert subsystem_spectral_measures(corr_empty)["subsystem_instability"] == 0.0
    assert subsystem_spectral_measures(corr_single)["subsystem_instability"] == 0.0

    assert graph_metrics(np.zeros((0, 0), dtype=float))["density"] == 0.0
    assert graph_metrics(np.zeros((1, 1), dtype=float))["density"] == 0.0


def test_process_frame_handles_empty_sensor_input_without_crash() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)

    last_result = {}
    for i in range(8):
        last_result = engine.process_frame(_frame(f"2024-01-01T00:00:0{i}Z", {}))

    assert last_result["structural_drift_score"] == 0.0
    assert last_result["relational_stability_score"] == 1.0
    assert "experimental_analytics" not in last_result


def test_process_frame_handles_single_sensor_input_without_relational_crash() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)

    result = {}
    for i in range(8):
        result = engine.process_frame(_frame(f"2024-01-01T00:01:0{i}Z", {"s1": float(i)}))

    assert "experimental_analytics" in result
    analytics = result["experimental_analytics"]
    assert analytics["directional"]["causal_divergence"] == 0.0
    assert analytics["subsystems"]["subsystem_instability"] == 0.0


def test_process_frame_handles_partially_missing_sensor_values() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)

    # Establish multi-signal baseline.
    for i in range(5):
        engine.process_frame(_frame(f"2024-01-01T00:02:0{i}Z", {"s1": float(i), "s2": float(i + 1)}))

    # One valid signal and one missing signal should not trigger relational failures.
    result = engine.process_frame(_frame("2024-01-01T00:02:09Z", {"s1": 9.0, "s2": None}))

    assert "experimental_analytics" in result
    analytics = result["experimental_analytics"]
    assert analytics["directional"]["causal_divergence"] == 0.0
    assert analytics["subsystems"]["subsystem_instability"] == 0.0
    assert isinstance(analytics["composite_instability"], float)
