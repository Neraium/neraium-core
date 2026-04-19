from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def lagged_correlation_matrix(observations: ArrayLike, lag: int = 1) -> np.ndarray:
    """Structural directional proxy C_ij = corr(x_i(t), x_j(t+lag)).

    Vectorized implementation using batch covariance for O(n²) instead of O(n²·T).
    """
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    if lag <= 0 or data.shape[0] <= lag:
        raise ValueError("Lag must be positive and smaller than the number of observations")

    current = np.nan_to_num(data[:-lag], nan=0.0, posinf=0.0, neginf=0.0)
    future = np.nan_to_num(data[lag:], nan=0.0, posinf=0.0, neginf=0.0)
    n_features = data.shape[1]

    # Vectorized correlation computation using standardization
    with np.errstate(invalid="ignore", divide="ignore"):
        # Normalize: (x - mean) / std
        current_mean = np.mean(current, axis=0, keepdims=True)
        current_std = np.std(current, axis=0, keepdims=True)
        current_std = np.where(current_std < 1e-12, 1e-12, current_std)
        current_norm = (current - current_mean) / current_std

        future_mean = np.mean(future, axis=0, keepdims=True)
        future_std = np.std(future, axis=0, keepdims=True)
        future_std = np.where(future_std < 1e-12, 1e-12, future_std)
        future_norm = (future - future_mean) / future_std

        # Compute correlation: mean(norm_x * norm_y) over time
        # Shape: (T, n_features) @ (n_features, n_features) = (T, n_features)
        # Then mean over T to get (n_features, n_features)
        n_obs = current.shape[0]
        result = (current_norm.T @ future_norm) / max(1.0, float(n_obs))

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
        "divergence": divergence,
        "proxy_only": 1.0,
    }
