from __future__ import annotations

from typing import Mapping


NumberLike = float | int


def composite_instability_score(
    components: Mapping[str, NumberLike | None],
    weights: Mapping[str, NumberLike] | None = None,
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
    weight_sum = 0.0

    for name, weight in active_weights.items():
        value = components.get(name)
        if value is None:
            continue

        value_f = float(value)
        if not (value_f == value_f):
            continue

        weight_f = float(weight)
        weighted_sum += value_f * weight_f
        weight_sum += weight_f

    if weight_sum <= 0.0:
        return 0.0

    return float(weighted_sum / weight_sum)
