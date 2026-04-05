"""
scoring.py — Compatibility shim for legacy weighted-average alert scoring.

STATUS: INTERFACE CONVENIENCE LAYER.  Not the internal model.

This module is preserved for backward compatibility with existing callers that
consume a scalar composite score from named structural components.  It is
NOT the system's internal representation of structural health.

The internal model is the structural operator 𝒮_t = (A_t, Σ_t).
See: neraium_core.operator.structural_score for the operator-derived alert path.

Data flow (correct):
    Observation window X(t)
        → fit_structural_operator(X)              [neraium_core.operator]
        → structural_metrics(current, baseline)   [neraium_core.operator.structural_score]
        → alert_score_from_operator(...)          [neraium_core.operator.structural_score]
        → display / alerting

Data flow (legacy, this module):
    {component: value, ...} dict
        → composite_instability_score_normalized(components, weights)
        → scalar composite score
        → display / alerting

The legacy path is valid for callers that lack a fitted StructuralOperator
(e.g. during warmup, or for components computed by legacy pipeline stages).
It should not be extended or treated as architecturally primary.

The DEFAULT_WEIGHTS below are calibration constants chosen empirically.
They are not derived from the operator model.  If you find yourself reasoning
about these weights as if they encode a theory of failure, stop and use
alert_score_from_operator() instead.
"""
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
    "transition_pressure": 0.0,
    "spectral": 0.0,
    "directional_divergence": 0.0,
    "entropy": 0.0,
    "subsystem_instability": 0.0,
    "early_warning": 0.0,
}


DEFAULT_WEIGHTS: dict[str, float] = {
    "relational_drift": 1.0,
    "regime_drift": 0.8,
    "transition_pressure": 0.95,
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

    def norm_key(key: object) -> str:
        return str(key).strip().lower()

    for key, value in values.items():
        k = norm_key(key)
        if k in DEFAULT_COMPONENTS or k in DEFAULT_WEIGHTS:
            normalized[k] = _coerce_float(value)

    for key, value in values.items():
        k = norm_key(key)
        mapped = LEGACY_KEYS.get(k, k)
        if mapped in normalized:
            continue
        if mapped in DEFAULT_COMPONENTS or mapped in DEFAULT_WEIGHTS:
            normalized[mapped] = _coerce_float(value)

    return normalized


def canonicalize_components(components: Mapping[str, object] | None = None) -> dict[str, float]:
    result = dict(DEFAULT_COMPONENTS)

    if not components:
        # Backward-compatible aliases
        result["drift"] = result["relational_drift"]
        result["directional"] = result["directional_divergence"]
        return result

    normalized = normalize_keys(components)
    result.update(normalized)
    # Backward-compatible aliases (tests and older callers may use these keys)
    result["drift"] = result.get("relational_drift", 0.0)
    result["directional"] = result.get("directional_divergence", 0.0)
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


# Winsorization cap for normalized composite (keeps scale compatible with decision thresholds)
DEFAULT_WINSORIZE_CAP = 3.0


def _winsorize(value: float, low: float = 0.0, high: float = DEFAULT_WINSORIZE_CAP) -> float:
    """Clip value to [low, high] for robust composite scoring."""
    if math.isnan(value) or math.isinf(value):
        return low
    return max(low, min(high, value))


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


def composite_instability_score_normalized(
    components: Mapping[str, object],
    weights: Mapping[str, object] | None = None,
    winsorize_cap: float = DEFAULT_WINSORIZE_CAP,
) -> float:
    """
    Composite instability score with robust per-component winsorization.
    Each component value is clipped to [0, winsorize_cap] before weighted average,
    so the result stays on a scale compatible with existing decision thresholds.
    """
    active = available_components(components, weights)

    if not active:
        return 0.0

    weighted_sum = sum(
        _winsorize(value, high=winsorize_cap) * weight for value, weight in active.values()
    )
    weight_sum = sum(weight for _, weight in active.values())

    if weight_sum <= 0.0:
        return 0.0

    return float(weighted_sum / weight_sum)
