from __future__ import annotations

import numpy as np


def directional_features(matrix: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    if safe.ndim != 2 or safe.shape[0] < 2:
        return np.zeros(safe.shape[1] if safe.ndim == 2 else 1, dtype=float)

    drift = safe[-1] - safe[0]
    signed_momentum = np.sum(np.diff(safe, axis=0), axis=0)
    positive_ratio = np.mean(np.diff(safe, axis=0) > 0.0, axis=0)
    return np.concatenate([drift, signed_momentum, positive_ratio]).astype(float)
