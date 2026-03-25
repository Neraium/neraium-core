from __future__ import annotations

import numpy as np


def residual_features(matrix: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    if safe.ndim != 2 or safe.shape[0] < 2:
        return np.zeros(safe.shape[1] if safe.ndim == 2 else 1, dtype=float)
    centered = safe - np.mean(safe, axis=0, keepdims=True)
    mad = np.mean(np.abs(centered), axis=0)
    std = np.std(safe, axis=0)
    return np.concatenate([mad, std]).astype(float)
