"""Fusion layer for AtomicMonitor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np


@dataclass
class FusionResult:
    total_score: float
    component_scores: Dict[str, float]
    dominant_detector: str
    weights_used: Dict[str, float]


class AtomicFusionEngine:
    """Weighted anomaly fusion with missing-output handling."""

    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights = {k: float(v) for k, v in weights.items()}

    def fuse(self, component_scores: Mapping[str, float | None]) -> FusionResult:
        available = {k: float(v) for k, v in component_scores.items() if v is not None and np.isfinite(v)}
        if not available:
            return FusionResult(0.0, {}, "none", {})

        raw_weights = {k: self.weights.get(k, 0.0) for k in available}
        z = sum(max(v, 0.0) for v in raw_weights.values())
        if z <= 0:
            norm = {k: 1.0 / len(available) for k in available}
        else:
            norm = {k: max(v, 0.0) / z for k, v in raw_weights.items()}

        total = float(sum(norm[k] * available[k] for k in available))
        total = float(np.clip(total, 0.0, 1.0))
        dominant = max(available, key=available.get)
        return FusionResult(total, dict(available), dominant, norm)
