"""Diagnostics helpers for atomic detector outputs."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def top_sensor_indices(scores: np.ndarray, k: int = 3) -> List[int]:
    if scores.size == 0:
        return []
    k = max(1, min(k, scores.size))
    return list(np.argsort(scores)[-k:][::-1])


def top_edges(matrix: np.ndarray, k: int = 3) -> List[Tuple[int, int, float]]:
    if matrix.ndim != 2:
        return []
    n = matrix.shape[0]
    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edges.append((i, j, float(abs(matrix[i, j]))))
    edges.sort(key=lambda x: x[2], reverse=True)
    return edges[:k]


def diagnostic_snapshot(total_score: float, components: Dict[str, float], dominant: str) -> Dict[str, object]:
    return {
        "total_score": float(total_score),
        "components": {k: float(v) for k, v in components.items()},
        "dominant_detector": dominant,
    }
