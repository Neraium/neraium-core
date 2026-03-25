from __future__ import annotations

import numpy as np


def temporal_features(matrix: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    if safe.ndim != 2 or safe.shape[0] < 2:
        return np.zeros(safe.shape[1] if safe.ndim == 2 else 1, dtype=float)

    diff = np.diff(safe, axis=0)
    mean_abs_velocity = np.mean(np.abs(diff), axis=0)
    acceleration = np.mean(np.abs(np.diff(diff, axis=0)), axis=0) if diff.shape[0] >= 2 else np.zeros(safe.shape[1])
    persistence = np.mean(np.sign(diff[1:]) == np.sign(diff[:-1]), axis=0) if diff.shape[0] >= 2 else np.zeros(safe.shape[1])
    return np.concatenate([mean_abs_velocity, acceleration, persistence]).astype(float)
