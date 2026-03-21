from __future__ import annotations

from .config import SIIConfig
from neraium_core.sii.types import (
    CONFIDENCE_LEVELS,
    INTERPRETED_STATES,
    STATES,
    ConfidenceLevel,
    InterpretedState,
    DecisionState,
)


def map_to_interpreted_state(
    *,
    structural_drift: float,
    relational_instability: float,
    regime_distance: float,
    coherence_score: float,
    graph_instability: float,
) -> InterpretedState:
    if (
        structural_drift >= 1.3
        and regime_distance >= 0.9
        and graph_instability >= 0.8
        and coherence_score < 0.45
    ):
        state: InterpretedState = "STRUCTURAL_INSTABILITY_OBSERVED"
    elif relational_instability >= 1.0 and graph_instability >= 0.9 and coherence_score < 0.5:
        state = "COUPLING_INSTABILITY_OBSERVED"
    elif regime_distance >= 0.85 and coherence_score >= 0.5 and graph_instability < 0.9:
        state = "REGIME_SHIFT_OBSERVED"
    elif coherence_score < 0.55 and structural_drift < 1.3:
        state = "COHERENCE_UNDER_CONSTRAINT"
    else:
        state = "NOMINAL_STRUCTURE"
    return safe_interpreted(state)


def map_decision_state(
    *,
    composite_score: float,
    interpreted_state: InterpretedState,
    trend: float,
    stability: float,
) -> DecisionState:
    """
    Read-only analytical mapping from structural evidence to STABLE/WATCH/ALERT.
    """
    score = float(composite_score)
    if interpreted_state in {"STRUCTURAL_INSTABILITY_OBSERVED", "COUPLING_INSTABILITY_OBSERVED"}:
        score += 0.15
    if trend > 0.03:
        score += 0.08
    if stability < 0.55:
        score += 0.08

    if score >= 1.45:
        state: DecisionState = "ALERT"
    elif score >= 0.80:
        state = "WATCH"
    else:
        state = "STABLE"

    if state not in STATES:
        return "STABLE"
    return state


def map_decision_state_with_config(
    *,
    composite_score: float,
    interpreted_state: InterpretedState,
    trend: float,
    stability: float,
    config: SIIConfig,
) -> DecisionState:
    """
    Config-aware decision mapping preserving read-only semantics.
    """
    score = float(composite_score)
    if interpreted_state in {"STRUCTURAL_INSTABILITY_OBSERVED", "COUPLING_INSTABILITY_OBSERVED"}:
        score += 0.15
    if trend > 0.03:
        score += 0.08
    if stability < 0.55:
        score += 0.08

    if score >= float(config.alert_threshold):
        return "ALERT"
    if score >= float(config.watch_threshold):
        return "WATCH"
    return "STABLE"


def safe_interpreted(value: str) -> InterpretedState:
    if value in INTERPRETED_STATES:
        return value  # type: ignore[return-value]
    return "NOMINAL_STRUCTURE"


def safe_confidence(value: str) -> ConfidenceLevel:
    if value in CONFIDENCE_LEVELS:
        return value  # type: ignore[return-value]
    return "low"
