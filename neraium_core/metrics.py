from __future__ import annotations

from typing import Mapping

from neraium_core.scoring import (
    canonicalize_components,
    canonicalize_weights,
    composite_instability_score_normalized,
)


def normalize_components(raw_components: Mapping[str, object] | None = None) -> dict[str, float]:
    """
    Canonical raw component vector.
    """
    return canonicalize_components(raw_components or {})


def calibrated_components(
    components: Mapping[str, object],
    component_confidence: Mapping[str, float],
) -> dict[str, float]:
    """
    Apply calibrated per-component confidence in [0,1].
    """
    canonical = canonicalize_components(components)
    out: dict[str, float] = {}
    for key, value in canonical.items():
        conf = float(component_confidence.get(key, 0.0))
        out[key] = float(value) * conf
    return out


def weighted_composite_score(
    components: Mapping[str, object],
    component_confidence: Mapping[str, float],
    base_weights: Mapping[str, object] | None = None,
) -> float:
    """
    One consistent scoring rule:
      raw components -> calibrated transforms -> weighted composite.
    """
    canonical = canonicalize_components(components)
    weights = canonicalize_weights(base_weights or None)
    calibrated_weights = {k: float(w) * float(component_confidence.get(k, 0.0)) for k, w in weights.items()}
    return float(composite_instability_score_normalized(canonical, weights=calibrated_weights))
