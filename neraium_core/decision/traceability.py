"""Decision traceability: lightweight explanation of why decisions are made.

Provides concise, human-readable trace of the top contributing factors
to a decision without verbose lists. Enhanced with causal abstraction,
temporal judgment, and uncertainty handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from neraium_core.decision.models import Decision
from neraium_core.decision import reasoning_upgrade as upgrade


@dataclass
class DecisionTrace:
    """Lightweight explanation of decision reasoning."""

    primary_factor: str  # Causal theme (max 50 chars)
    secondary_factors: list[str]  # Temporal phase + trajectory + persistence (max 3, 30 chars each)
    confidence_rationale: str  # Uncertainty-aware brief rationale


def build_decision_trace(decision: Decision) -> Optional[DecisionTrace]:
    """Extract key factors from a Decision into a concise trace.

    Uses causal abstraction, temporal judgment, and uncertainty handling
    to produce human-readable reasoning.

    Args:
        decision: Decision to extract trace from

    Returns:
        DecisionTrace with primary causal theme and secondary factors
    """
    # Extract metrics needed for reasoning upgrade
    drift = _estimate_drift_from_decision(decision)
    relational_stability = _estimate_relational_stability(decision)
    coherence = _estimate_coherence(decision)
    drift_velocity = decision.temporal_context.drift_velocity if decision.temporal_context else 0.0
    coherence_trend = decision.temporal_context.confidence_trend if decision.temporal_context else 0.0

    # Step 1: Derive causal theme
    causal_theme = upgrade.derive_causal_theme(
        drift=drift,
        relational_stability=relational_stability,
        coherence=coherence,
        persistence_frames=decision.persistence_frames_at_level,
        trajectory=decision.trajectory,
    )

    # Step 2: Classify temporal phase
    temporal_phase = upgrade.classify_temporal_phase(
        persistence_frames=decision.persistence_frames_at_level,
        drift_velocity=drift_velocity,
        trajectory=decision.trajectory,
        coherence_trend=coherence_trend,
    )

    # Step 3: Classify uncertainty context
    pattern_match_tier = decision.pattern_match_tier if decision.pattern_match else None
    causal_chain_strength = _estimate_causal_chain_strength(decision)
    uncertainty_context = upgrade.classify_uncertainty_context(
        finding_confidence=decision.finding_confidence,
        persistence_frames=decision.persistence_frames_at_level,
        trajectory=decision.trajectory,
        pattern_match_tier=pattern_match_tier,
        causal_chain_strength=causal_chain_strength,
    )

    # Format primary factor using causal theme
    primary = upgrade.format_causal_summary(causal_theme)

    # Format secondary factors: temporal phase + trajectory + persistence
    secondary = []
    temporal_desc = upgrade.format_temporal_context(temporal_phase, decision.persistence_frames_at_level)
    if temporal_desc:
        secondary.append(temporal_desc)

    # Add trajectory if different from implicit in temporal phase
    if decision.trajectory == "unstable":
        secondary.append("Trajectory unstable")

    # Limit to 3 secondary factors
    secondary = secondary[:3]

    # Format confidence rationale using uncertainty context
    confidence = upgrade.format_uncertainty_rationale(
        uncertainty_context,
        decision.finding_confidence,
        decision.persistence_frames_at_level,
    )

    return DecisionTrace(
        primary_factor=primary,
        secondary_factors=secondary,
        confidence_rationale=confidence,
    )


def _estimate_drift_from_decision(decision: Decision) -> float:
    """Estimate structural drift from decision data."""
    # Use findings to infer drift magnitude
    if decision.findings:
        confidence_sum = sum(f.magnitude for f in decision.findings if f.magnitude)
        if confidence_sum > 0:
            return min(1.0, confidence_sum / len(decision.findings))

    # Use causal chain as proxy
    if decision.causal_chain and decision.causal_chain.confidence:
        return min(1.0, decision.causal_chain.confidence)

    # Estimate from severity and confidence
    severity_map = {"LOW": 0.0, "MODERATE": 0.3, "ELEVATED": 0.6, "HIGH": 0.9}
    base = severity_map.get(decision.severity, 0.0)
    return min(1.0, base * decision.finding_confidence)


def _estimate_relational_stability(decision: Decision) -> float:
    """Estimate relational stability from decision data."""
    # Use causal chain strength as proxy for relational stability
    chain_strength = _estimate_causal_chain_strength(decision)

    # Also consider pattern match quality
    if decision.pattern_match:
        return min(1.0, max(chain_strength, decision.pattern_match.confidence))

    return chain_strength


def _estimate_coherence(decision: Decision) -> float:
    """Estimate system coherence from decision data."""
    # High coherence when: few coherence errors, consistent trajectory, no contradiction
    base_coherence = 0.7  # Assume good coherence by default

    if decision.coherence_errors:
        # Reduce coherence for each error
        base_coherence -= 0.1 * min(3, len(decision.coherence_errors))

    # Unstable trajectory reduces coherence
    if decision.trajectory == "unstable":
        base_coherence -= 0.2

    # Improving coherence with high confidence
    if decision.finding_confidence > 0.8 and decision.trajectory in {"stable", "recovering"}:
        base_coherence = min(1.0, base_coherence + 0.1)

    return max(0.0, min(1.0, base_coherence))


def _estimate_causal_chain_strength(decision: Decision) -> float:
    """Estimate causal chain strength from decision data."""
    if decision.causal_chain is None:
        return 0.3

    chain_conf = float(decision.causal_chain.confidence or 0.5)
    steps_count = len(decision.causal_chain.steps or [])

    # Strength increases with confidence and number of well-formed steps
    strength = chain_conf
    if steps_count >= 2:
        strength = min(1.0, strength + 0.15)
    if steps_count >= 3:
        strength = min(1.0, strength + 0.1)

    return strength


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
