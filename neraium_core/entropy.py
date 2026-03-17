from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def interaction_entropy(matrix: ArrayLike) -> float:
    values = np.abs(np.asarray(matrix, dtype=float)).ravel()
    total = float(values.sum())
    if total <= 0:
        return 0.0

    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())
