from __future__ import annotations

from typing import Mapping

import math


LEGACY_KEYS: dict[str, str] = {
    "drift": "relational_drift",
    "directional": "directional_divergence",
    "causal": "directional_divergence",
    "subsystem": "subsystem_instability",
    "subsystem_max": "subsystem_instability",
}


DEFAULT_COMPONENTS: dict[str, float] = {
    "relational_drift": 0.0,
    "regime_drift": 0.0,
    "spectral": 0.0,
    "directional_divergence": 0.0,
    "entropy": 0.0,
    "subsystem_instability": 0.0,
    "early_warning": 0.0,
}


DEFAULT_WEIGHTS: dict[str, float] = {
    "relational_drift": 1.0,
    "regime_drift": 0.8,
    "spectral": 0.8,
    "directional_divergence": 0.8,
    "entropy": 0.5,
    "subsystem_instability": 0.7,
    "early_warning": 0.6,
}


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)  # type: ignore[arg-type]
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def normalize_keys(values: Mapping[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}

    for key, value in values.items():
        if key in DEFAULT_COMPONENTS or key in DEFAULT_WEIGHTS:
            normalized[key] = _coerce_float(value)

    for key, value in values.items():
        mapped = LEGACY_KEYS.get(key, key)
        if mapped in normalized:
            continue
        normalized[mapped] = _coerce_float(value)

    return normalized


def canonicalize_components(components: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_COMPONENTS)

    if not components:
        return result

    normalized = normalize_keys(components)
    result.update(normalized)
    return result


def canonicalize_weights(weights: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_WEIGHTS)

    if not weights:
        return result

    normalized = normalize_keys(weights)
    result.update(normalized)
    return result


def available_components(
    components: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
) -> dict[str, tuple[float, float]]:
    canonical_components = canonicalize_components(components)
    canonical_weights = canonicalize_weights(weights)

    active: dict[str, tuple[float, float]] = {}
    for key, value in canonical_components.items():
        weight = canonical_weights.get(key, 0.0)
        if weight > 0.0:
            active[key] = (value, weight)

    return active


def composite_instability_score(
    components: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
    normalize: bool = True,
) -> float:
    active = available_components(components, weights)

    if not active:
        return 0.0

    weighted_sum = sum(value * weight for value, weight in active.values())
    weight_sum = sum(weight for _, weight in active.values())

    if not normalize or weight_sum <= 0.0:
        return float(weighted_sum)

    return float(weighted_sum / weight_sum)