"""Incremental rolling statistics aligned with batch definitions where possible."""

from __future__ import annotations

import numpy as np

from neraium_core.realtime.numerics import EPS, sanitize_finite


def rolling_mean_var_last_row(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean and variance (population) over rows."""
    x = sanitize_finite(np.asarray(data, dtype=np.float64))
    if x.ndim != 2 or x.shape[0] < 1:
        return np.array([]), np.array([])
    m = np.mean(x, axis=0)
    v = np.var(x, axis=0)
    return m, v


def early_warning_metrics_incremental(observations: np.ndarray) -> dict[str, float]:
    """
    Same contract as neraium_core.early_warning.early_warning_metrics
    but operates on a pre-built 2D array (recent window only).
    """
    data = sanitize_finite(np.asarray(observations, dtype=np.float64))
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    safe = data
    variances = np.var(safe, axis=0)
    if safe.shape[0] < 2:
        lag1 = np.zeros(safe.shape[1], dtype=float)
    else:
        lag1 = []
        for idx in range(safe.shape[1]):
            a = safe[:-1, idx]
            b = safe[1:, idx]
            den = float(np.std(a) * np.std(b))
            if den < EPS:
                lag1.append(0.0)
            else:
                c = float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / den)
                if not np.isfinite(c):
                    c = 0.0
                lag1.append(c)
        lag1 = np.asarray(lag1, dtype=float)

    return {
        "variance": float(np.mean(variances) if variances.size else 0.0),
        "lag1_autocorrelation": float(np.mean(lag1) if lag1.size else 0.0),
    }
