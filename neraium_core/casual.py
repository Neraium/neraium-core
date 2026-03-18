from __future__ import annotations

import numpy as np


def granger_causality_matrix(X: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Lightweight Granger-style proxy matrix.

    This is not formal causal proof. It estimates directional influence using
    lagged univariate regression quality as a structural proxy.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] <= lag or X.shape[1] < 2:
        return np.zeros((X.shape[1] if X.ndim == 2 else 0, X.shape[1] if X.ndim == 2 else 0))

    n = X.shape[1]
    C = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            x = X[:-lag, i]
            y = X[lag:, j]

            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            if len(x) < 5:
                continue

            denom = float(np.dot(x, x))
            if abs(denom) < 1e-12:
                continue

            beta = float(np.dot(x, y) / denom)
            pred = beta * x
            error = float(np.mean((y - pred) ** 2))

            C[i, j] = 1.0 / (error + 1e-6)

    return C


def causal_metrics(C: np.ndarray) -> dict[str, float]:
    """Compute causal-proxy summary metrics."""
    C = np.asarray(C, dtype=float)

    if C.size == 0:
        return {
            "energy": 0.0,
            "asymmetry": 0.0,
            "divergence": 0.0,
            "causal_energy": 0.0,
            "causal_asymmetry": 0.0,
            "causal_divergence": 0.0,
        }

    energy = float(np.mean(np.abs(C)))
    asymmetry = float(np.mean(np.abs(C - C.T)))
    divergence = float(energy * (1.0 + asymmetry))

    return {
        "energy": energy,
        "asymmetry": asymmetry,
        "divergence": divergence,
        "causal_energy": energy,
        "causal_asymmetry": asymmetry,
        "causal_divergence": divergence,
    }