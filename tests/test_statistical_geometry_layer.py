from __future__ import annotations

import numpy as np

from neraium_core.stat_geometry.geometry_integration import StatisticalGeometryLayer


def _matrix(t: int) -> np.ndarray:
    rows = []
    for i in range(12):
        x = 0.02 * (t + i)
        rows.append([x, 0.8 * x + 0.1 * np.sin(0.1 * (t + i)), -0.4 * x + 0.05 * np.cos(0.12 * (t + i))])
    return np.asarray(rows, dtype=float)


def test_geometry_layer_populates_expected_blocks() -> None:
    layer = StatisticalGeometryLayer(max_history=64, graph_window=12, stats_window=10)
    out = {}
    for t in range(28):
        out = layer.update(entity_id="asset-A", matrix=_matrix(t), representation_mode="combined")

    assert "geometry" in out
    assert "state_space_statistics" in out
    assert "state_graph" in out
    assert "geometry_explanations" in out
    assert out["geometry"]["path_length"] >= 0.0
    assert out["state_space_statistics"]["local_density"] >= 0.0
    assert out["state_graph"]["node_count"] >= 1


def test_geometry_layer_fleet_summary_multiple_entities() -> None:
    layer = StatisticalGeometryLayer(max_history=64)
    for t in range(24):
        layer.update(entity_id="asset-A", matrix=_matrix(t), representation_mode="dynamic")
        layer.update(entity_id="asset-B", matrix=_matrix(t + 8) + 0.5, representation_mode="dynamic")
    out = layer.update(entity_id="asset-C", matrix=_matrix(33) - 0.2, representation_mode="dynamic")
    fleet = out["fleet_geometry"]
    assert fleet["fleet_entity_count"] >= 3
    assert fleet["fleet_mean_pairwise_distance"] >= 0.0


def test_geometry_layer_waits_until_min_history_is_available() -> None:
    layer = StatisticalGeometryLayer(max_history=64, graph_window=8, stats_window=8, min_history=20)
    out = {}
    for t in range(10):
        out = layer.update(entity_id="asset-A", matrix=_matrix(t), representation_mode="combined")
    assert out["geometry"]["available"] is False
    assert out["state_space_statistics"]["available"] is False
