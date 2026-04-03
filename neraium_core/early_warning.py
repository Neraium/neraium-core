from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def early_warning_metrics(observations: ArrayLike) -> dict[str, float]:
    data = np.asarray(observations, dtype=float)
    if data.ndim != 2:
        raise ValueError("Expected 2D observations")
    safe = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    variances = np.var(safe, axis=0)
    if safe.shape[0] < 2:
        lag1 = np.zeros(safe.shape[1], dtype=float)
    else:
        lag1 = []
        for idx in range(safe.shape[1]):
            lag1.append(np.corrcoef(safe[:-1, idx], safe[1:, idx])[0, 1])
        lag1 = np.nan_to_num(np.array(lag1, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "variance": float(np.mean(variances) if variances.size else 0.0),
        "lag1_autocorrelation": float(np.mean(lag1) if lag1.size else 0.0),
    }
