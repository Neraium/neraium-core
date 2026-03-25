from __future__ import annotations

import numpy as np


def multiscale_features(matrix: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    if safe.ndim != 2 or safe.shape[0] < 3:
        width = safe.shape[1] if safe.ndim == 2 else 1
        return np.zeros(width * 6, dtype=float)

    windows = [max(2, safe.shape[0] // 6), max(2, safe.shape[0] // 3), max(2, safe.shape[0] // 2)]
    out: list[float] = []
    for w in windows:
        tail = safe[-w:]
        out.extend(np.mean(tail, axis=0).tolist())
        out.extend(np.std(tail, axis=0).tolist())
    return np.asarray(out, dtype=float)
