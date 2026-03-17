from __future__ import annotations

from typing import Mapping


DEFAULT_WEIGHTS = {
    "relational_drift": 0.2,
    "spectral": 0.16,
    "directional_divergence": 0.14,
    "entropy": 0.1,
    "early_warning": 0.14,
    "subsystem_instability": 0.16,
    "forecast": 0.1,
}

LEGACY_KEYS = {
    "drift": "relational_drift",
    "directional": "directional_divergence",
}


def _normalized_weights(weights: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 0:
        return {key: 0.0 for key in weights}
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def composite_instability_score(
    components: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
) -> float:
    active_weights = _normalized_weights(weights if weights is not None else DEFAULT_WEIGHTS)

    normalized_components: dict[str, float] = {}
    for key, value in components.items():
        mapped = LEGACY_KEYS.get(key, key)
        normalized_components[mapped] = float(value)

    return float(
        sum(
            normalized_components.get(name, 0.0) * weight
            for name, weight in active_weights.items()
        )
    )
