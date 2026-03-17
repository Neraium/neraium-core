from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def lagged_correlation_matrix(observations: ArrayLike, lag: int = 1) -> np.ndarray:
    """Structural directional proxy C_ij = corr(x_i(t), x_j(t+lag))."""
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    if lag <= 0 or data.shape[0] <= lag:
        raise ValueError("Lag must be positive and smaller than the number of observations")

    current = np.nan_to_num(data[:-lag], nan=0.0, posinf=0.0, neginf=0.0)
    future = np.nan_to_num(data[lag:], nan=0.0, posinf=0.0, neginf=0.0)
    n_features = data.shape[1]
    result = np.zeros((n_features, n_features), dtype=float)

    for i in range(n_features):
        for j in range(n_features):
            result[i, j] = np.corrcoef(current[:, i], future[:, j])[0, 1]

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def directional_metrics(matrix: ArrayLike) -> dict[str, float]:
    lagged = np.asarray(matrix, dtype=float)
    if lagged.ndim != 2 or lagged.shape[0] != lagged.shape[1]:
        raise ValueError("Directional matrix must be square")

    abs_matrix = np.abs(lagged)
    energy = float(np.mean(abs_matrix))
    asymmetry = float(np.mean(np.abs(lagged - lagged.T)))
    divergence = float(energy * (1.0 + asymmetry))

    return {
        "causal_energy": energy,
        "causal_asymmetry": asymmetry,
        "causal_divergence": divergence,
        "proxy_only": 1.0,
    }
