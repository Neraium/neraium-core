"""Decision traceability: lightweight explanation of why decisions are made.

Provides concise, human-readable trace of the top contributing factors
to a decision without verbose lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from neraium_core.decision.models import Decision


@dataclass
class DecisionTrace:
    """Lightweight explanation of decision reasoning."""

    primary_factor: str  # Top contributing signal (max 50 chars)
    secondary_factors: list[str]  # 1-2 supporting factors (max 30 chars each)
    confidence_rationale: str  # Brief rationale for confidence level


def build_decision_trace(decision: Decision) -> Optional[DecisionTrace]:
    """Extract key factors from a Decision into a concise trace.

    Args:
        decision: Decision to extract trace from

    Returns:
        DecisionTrace with primary and secondary factors
    """
    primary = _extract_primary_factor(decision)
    secondary = _extract_secondary_factors(decision)
    confidence = _extract_confidence_rationale(decision)

    if not primary:
        return None

    return DecisionTrace(
        primary_factor=primary,
        secondary_factors=secondary,
        confidence_rationale=confidence,
    )


def _extract_primary_factor(decision: Decision) -> Optional[str]:
    """Extract the single most important contributing factor."""
    # Priority 1: Stage transitions are always primary
    if decision.stage_transition_event:
        return f"Stage change: {decision.stage_transition_event}"

    # Priority 2: HIGH severity is primary
    if decision.severity == "HIGH":
        return f"HIGH severity detected"

    # Priority 3: Degradation stage progression
    if decision.degradation_stage in {
        "failure_approach",
        "accelerated_deterioration",
        "persistent_degradation",
    }:
        return f"{decision.degradation_stage.replace('_', ' ').title()}"

    # Priority 4: Pattern match with outcome
    if decision.pattern_match and decision.pattern_match_tier in {"moderate", "strong"}:
        if decision.pattern_outcome_type == "failure_progression":
            return f"Failure-like pattern match (tier: {decision.pattern_match_tier})"
        elif decision.pattern_outcome_type == "persistent_degradation":
            return f"Persistent degradation pattern"

    # Priority 5: Confidence and finding signal
    if decision.finding_confidence > 0.85:
        return "High-confidence structural change"

    if decision.trajectory == "degrading":
        return "Degrading trajectory detected"

    if decision.temporal_confidence_delta > 0.3:
        return "Confidence increasing"

    # Default
    if decision.severity != "LOW":
        return f"{decision.severity} severity with {decision.trajectory} trajectory"

    return "Baseline stable state"


def _extract_secondary_factors(decision: Decision) -> list[str]:
    """Extract 1-2 supporting factors."""
    factors = []

    # Factor 1: Persistence
    if decision.persistence_frames_at_level > 3:
        persistence = f"Persisted {decision.persistence_frames_at_level} frames"
        factors.append(persistence)

    # Factor 2: Pattern outcome influence
    if (
        decision.pattern_outcome_type
        and decision.pattern_match_tier in {"moderate", "strong"}
    ):
        factors.append(
            f"Pattern: {decision.pattern_outcome_type.replace('_', ' ').title()}"
        )

    # Factor 3: Temporal confidence trend
    if decision.temporal_confidence_delta > 0.2:
        factors.append(f"Confidence trend up")
    elif decision.temporal_confidence_delta < -0.2:
        factors.append(f"Confidence trend down")

    # Factor 4: Transient likelihood
    if decision.transient_score > 0.7 and not decision.suppress:
        factors.append("Likely transient")

    # Factor 5: First appearance
    if decision.is_first_appearance and decision.severity in {"ELEVATED", "MODERATE"}:
        factors.append("First appearance")

    # Return top 2
    return factors[:2]


def _extract_confidence_rationale(decision: Decision) -> str:
    """Extract brief rationale for confidence level."""
    avg_confidence = (decision.finding_confidence + decision.action_confidence) / 2.0

    if avg_confidence > 0.8:
        if (
            decision.pattern_match
            and decision.pattern_match_tier == "strong"
        ):
            return "High confidence (multi-signal, pattern match)"
        elif decision.persistence_frames_at_level > 5:
            return "High confidence (sustained signal)"
        else:
            return "High confidence (strong signal)"

    elif avg_confidence > 0.6:
        return "Moderate confidence (confirmed signal)"

    elif avg_confidence > 0.4:
        return "Emerging confidence (early stage)"

    else:
        if decision.suppress:
            return "Low confidence, suppressed"
        return "Low confidence (monitor)"


def format_trace_for_display(trace: Optional[DecisionTrace]) -> dict[str, Any]:
    """Format trace for operator display.

    Args:
        trace: DecisionTrace to format

    Returns:
        Dictionary suitable for JSON serialization
    """
    if not trace:
        return {}

    return {
        "decision_trace": {
            "primary_factor": trace.primary_factor,
            "secondary_factors": trace.secondary_factors,
            "confidence_rationale": trace.confidence_rationale,
        }
    }


def validate_trace(trace: Optional[DecisionTrace]) -> tuple[bool, list[str]]:
    """Validate trace field lengths and content.

    Args:
        trace: Trace to validate

    Returns:
        (is_valid, error_messages)
    """
    if not trace:
        return True, []

    errors = []

    if len(trace.primary_factor) > 60:
        errors.append(f"primary_factor too long: {len(trace.primary_factor)} chars")

    for i, factor in enumerate(trace.secondary_factors):
        if len(factor) > 40:
            errors.append(f"secondary_factors[{i}] too long: {len(factor)} chars")

    if len(trace.confidence_rationale) > 50:
        errors.append(f"confidence_rationale too long: {len(trace.confidence_rationale)} chars")

    return len(errors) == 0, errors
