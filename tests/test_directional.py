from __future__ import annotations

import numpy as np

from neraium_core.directional import directional_metrics, lagged_correlation_matrix


def test_directional_metrics() -> None:
    observations = np.array(
        [
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
        ]
    )
    matrix = lagged_correlation_matrix(observations, lag=1)
    metrics = directional_metrics(matrix)
    assert metrics["causal_energy"] >= 0.0
    assert metrics["causal_divergence"] >= metrics["causal_energy"]
