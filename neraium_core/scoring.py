from __future__ import annotations

from typing import Mapping


CANONICAL_COMPONENTS = (
    "drift",
    "spectral",
    "directional",
    "entropy",
    "early_warning",
    "subsystem_instability",
)


def canonicalize_components(components: Mapping[str, float]) -> dict[str, float]:
    canonical = {name: 0.0 for name in CANONICAL_COMPONENTS}
    for name, value in components.items():
        key = name.strip().lower()
        if key in canonical:
            canonical[key] = float(value)
    return canonical


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
    canonical = canonicalize_components(components)
    return float(sum(canonical.get(name, 0.0) * float(weight) for name, weight in active_weights.items()))
