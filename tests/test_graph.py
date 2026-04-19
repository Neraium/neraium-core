from __future__ import annotations

import numpy as np

from neraium_core.graph import graph_metrics, thresholded_adjacency


def test_graph_metrics_include_connectivity() -> None:
    corr = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.9], [0.1, 0.9, 1.0]])
    adjacency = thresholded_adjacency(corr, threshold=0.6)
    metrics = graph_metrics(adjacency, corr=corr)
    assert metrics["mean_degree"] >= 0.0
    assert metrics["density"] >= 0.0
    assert metrics["clustering"] >= 0.0
    assert metrics["connectivity"] in (0.0, 1.0)
    assert metrics["mean_absolute_connectivity"] >= 0.0
