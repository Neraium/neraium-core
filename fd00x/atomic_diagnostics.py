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


def diagnostic_snapshot(
    total_score: float,
    components: Dict[str, float],
    dominant: str,
    smoothed_score: float = 0.0,
    raw_component_scores: Dict[str, float] = None,
    normalized_component_scores: Dict[str, float] = None,
    active_detector_count: int = 0,
) -> Dict[str, object]:
    """Return a comprehensive diagnostic snapshot for one timestep.

    Includes raw component scores, normalized component scores, the smoothed
    fused score, and the number of active detectors — the key fields needed
    to diagnose false positive behaviour.
    """
    return {
        "total_score": float(total_score),
        "smoothed_score": float(smoothed_score),
        "active_detector_count": int(active_detector_count),
        "components": {k: float(v) for k, v in components.items()},
        "raw_component_scores": {k: float(v) for k, v in (raw_component_scores or {}).items()},
        "normalized_component_scores": {k: float(v) for k, v in (normalized_component_scores or {}).items()},
        "dominant_detector": dominant,
    }
