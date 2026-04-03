from __future__ import annotations

import numpy as np

from neraium_core.stat_geometry.state_graph import compute_state_graph


def test_state_graph_handles_mixed_dimension_history_without_crashing() -> None:
    path = [
        np.array([0.0, 0.1, 0.2]),
        np.array([0.1, 0.2, 0.3, 7.0]),
        np.array([0.2, 0.3, 0.4]),
        np.array([0.3, 0.5, 0.7, 6.0, -2.0]),
        np.array([0.5, 0.8, 1.0]),
        np.array([0.7, 1.0, 1.3, 1.0]),
        np.array([0.9, 1.3, 1.7]),
        np.array([1.2, 1.6, 2.0, 0.5]),
    ]

    first = compute_state_graph(path, window=6)
    second = compute_state_graph(path, window=6)

    assert first == second
    assert set(first.keys()) == {
        "node_count",
        "edge_count",
        "branching_factor",
        "branching_intensity",
        "transition_entropy",
        "transition_entropy_effective",
        "revisit_rate",
        "path_commitment_score",
        "graph_divergence_score",
        "graph_divergence_effective",
        "directional_inconsistency",
        "raw_divergence_score",
        "graph_density",
        "region_histogram",
    }
    assert isinstance(first["region_histogram"], dict)
    assert first["node_count"] >= 1
    assert first["edge_count"] >= 1
