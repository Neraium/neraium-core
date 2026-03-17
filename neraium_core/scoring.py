from __future__ import annotations

from typing import Mapping


def composite_instability_score(
    components: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    default_weights = {
        "drift": 0.22,
        "spectral": 0.18,
        "directional": 0.16,
        "entropy": 0.12,
        "early_warning": 0.14,
        "subsystem_instability": 0.18,
    }
    active_weights = dict(weights) if weights is not None else default_weights
    return float(sum(float(components.get(name, 0.0)) * weight for name, weight in active_weights.items()))
