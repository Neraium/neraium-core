from __future__ import annotations

from typing import Mapping


def composite_instability_score(
    components: Mapping[str, float | None],
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

    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in active_weights.items():
        value = components.get(name)
        if value is None:
            continue
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)

    if total_weight == 0.0:
        return 0.0

    return float(weighted_sum / total_weight)
