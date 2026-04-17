"""Confidence scoring for findings and actions.

Separates "do we think something happened" from "do we know what to do about it".
"""

from __future__ import annotations

from typing import Any


def score_finding_confidence(
    *,
    drift_score: float,
    signal_count: int,
    data_quality_issues: int,
    state: str,
    relational_instability: float,
) -> float:
    """Score confidence that something genuinely changed (vs noise/transient).

    Returns [0, 1] where 1 = very confident.
    """
    base_confidence = 0.5

    if drift_score > 0.5:
        base_confidence += 0.3
    elif drift_score > 0.3:
        base_confidence += 0.15

    if relational_instability > 0.4:
        base_confidence += 0.15

    signal_quality = 1.0 - (data_quality_issues / max(1, signal_count))
    base_confidence *= signal_quality

    if state in {"ALERT", "WATCH"}:
        base_confidence = min(1.0, base_confidence + 0.2)

    return min(1.0, max(0.0, base_confidence))


def score_action_confidence(
    *,
    finding_confidence: float,
    causal_chain_strength: float,
    pattern_match_confidence: float,
    recommendation_clarity: float,
) -> float:
    """Score confidence in the recommended action.

    Returns [0, 1] where 1 = very confident in the action.
    High action confidence means the recommendation is well-grounded and likely useful.
    """
    base = finding_confidence * 0.4

    if causal_chain_strength > 0.6:
        base += 0.3
    elif causal_chain_strength > 0.3:
        base += 0.15

    if pattern_match_confidence > 0.7:
        base += 0.2

    base += recommendation_clarity * 0.1

    return min(1.0, max(0.0, base))


def categorize_confidence(score: float) -> str:
    """Convert [0, 1] confidence to human-readable label."""
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def should_surface_finding(
    *,
    severity: str,
    suppress_flag: bool,
    transient_score: float,
    finding_confidence: float,
) -> bool:
    """Determine if we should show this to the operator.

    Rules:
    - CRITICAL severity is NEVER suppressed
    - HIGH/MODERATE need decent confidence
    - LOW severity can be suppressed if transient
    """
    if severity == "CRITICAL":
        return True

    if suppress_flag and severity not in {"CRITICAL", "HIGH"}:
        return False

    if severity == "HIGH":
        return finding_confidence >= 0.5

    if severity == "MODERATE":
        return finding_confidence >= 0.6 and transient_score < 0.7

    if severity == "LOW":
        return finding_confidence >= 0.7 and transient_score < 0.8

    return True
