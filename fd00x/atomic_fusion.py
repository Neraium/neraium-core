"""Fusion layer for AtomicMonitor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np


@dataclass
class FusionResult:
    total_score: float
    component_scores: Dict[str, float]
    dominant_detector: str
    weights_used: Dict[str, float]
    active_count: int = 0


class AtomicFusionEngine:
    """Weighted anomaly fusion with missing-output handling and conservative gating.

    Gating rule: if fewer than ``min_active`` components exceed ``activation_floor``,
    the total fused score is multiplied by ``downweight_factor``.  This prevents a
    single noisy component from driving RED behaviour.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        activation_floor: float = 0.15,
        min_active: int = 2,
        downweight_factor: float = 0.3,
    ) -> None:
        self.weights = {k: float(v) for k, v in weights.items()}
        self.activation_floor = float(activation_floor)
        self.min_active = int(min_active)
        self.downweight_factor = float(downweight_factor)

    def fuse(self, component_scores: Mapping[str, float | None]) -> FusionResult:
        # Clip each finite component score to [0, 1] before fusion
        available = {
            k: float(np.clip(v, 0.0, 1.0))
            for k, v in component_scores.items()
            if v is not None and np.isfinite(v)
        }
        if not available:
            return FusionResult(0.0, {}, "none", {}, active_count=0)

        raw_weights = {k: self.weights.get(k, 0.0) for k in available}
        z = sum(max(v, 0.0) for v in raw_weights.values())
        if z <= 0:
            norm = {k: 1.0 / len(available) for k in available}
        else:
            norm = {k: max(v, 0.0) / z for k, v in raw_weights.items()}

        total = float(sum(norm[k] * available[k] for k in available))
        total = float(np.clip(total, 0.0, 1.0))

        # Count components above activation floor
        active_count = sum(1 for v in available.values() if v >= self.activation_floor)

        # Downweight if not enough components are active
        if active_count < self.min_active:
            total *= self.downweight_factor

        dominant = max(available, key=available.get)
        return FusionResult(total, dict(available), dominant, norm, active_count=active_count)
