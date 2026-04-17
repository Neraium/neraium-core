"""Output normalization for consistent numeric formatting.

Ensures all numeric fields are properly rounded and formatted according
to their semantic meaning (confidence, score, percentage, integer).
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import Decision


def normalize_confidence(value: float) -> float:
    """Normalize confidence [0,1] to 2 decimal places."""
    return round(max(0.0, min(1.0, float(value))), 2)


def normalize_score(value: float) -> float:
    """Normalize score [0,3] to 3 decimal places."""
    return round(float(value), 3)


def normalize_percentage(value: float) -> float:
    """Normalize percentage to 1 decimal place."""
    return round(float(value), 1)


def normalize_integer(value: Any) -> int:
    """Normalize to integer."""
    return int(value)


def normalize_decision_output(decision: Decision) -> dict[str, Any]:
    """Normalize a Decision to serializable dict with proper formatting.

    Args:
        decision: Decision object to normalize

    Returns:
        Dictionary with normalized numeric values
    """
    output = decision.to_dict()

    # Normalize confidence metrics
    if "finding_confidence" in output:
        output["finding_confidence"] = normalize_confidence(output["finding_confidence"])
    if "action_confidence" in output:
        output["action_confidence"] = normalize_confidence(output["action_confidence"])
    if "transient_score" in output:
        output["transient_score"] = normalize_confidence(output["transient_score"])

    # Normalize temporal context if present
    if output.get("temporal_context"):
        tc = output["temporal_context"]
        if "drift_velocity" in tc:
            tc["drift_velocity"] = normalize_score(tc["drift_velocity"])
        if "confidence_trend" in tc:
            tc["confidence_trend"] = normalize_score(tc["confidence_trend"])
        if "drift_scores" in tc and isinstance(tc["drift_scores"], list):
            tc["drift_scores"] = [normalize_score(s) for s in tc["drift_scores"]]
        if "confidence_scores" in tc and isinstance(tc["confidence_scores"], list):
            tc["confidence_scores"] = [normalize_confidence(s) for s in tc["confidence_scores"]]

    # Normalize pattern match confidence if present
    if output.get("pattern_match"):
        pm = output["pattern_match"]
        if "confidence" in pm:
            pm["confidence"] = normalize_confidence(pm["confidence"])
        if "similarity" in pm:
            pm["similarity"] = normalize_confidence(pm["similarity"])

    # Normalize persistence frames (integers)
    if "persistence_frames_at_level" in output:
        output["persistence_frames_at_level"] = normalize_integer(
            output["persistence_frames_at_level"]
        )

    # Normalize persistence state if present
    if output.get("persistence_state"):
        ps = output["persistence_state"]
        if "consecutive_frames_at_level" in ps and isinstance(ps["consecutive_frames_at_level"], dict):
            ps["consecutive_frames_at_level"] = {
                k: normalize_integer(v) for k, v in ps["consecutive_frames_at_level"].items()
            }
        if "suppression_streak" in ps:
            ps["suppression_streak"] = normalize_integer(ps["suppression_streak"])

    # Ensure temporal_confidence_delta is normalized
    if "temporal_confidence_delta" in output:
        output["temporal_confidence_delta"] = normalize_score(
            output["temporal_confidence_delta"]
        )

    return output


def format_for_operator_display(decision: Decision) -> dict[str, Any]:
    """Format decision for operator-facing display.

    Removes internal fields and normalizes all numeric values.

    Args:
        decision: Decision to format

    Returns:
        Operator-facing dict with normalized values
    """
    normalized = normalize_decision_output(decision)

    # Remove internal fields not needed for operators
    internal_fields = {
        "temporal_context",  # Detailed history, not for display
        "persistence_state",  # Internal tracking
        "consistency_check_passed",  # Internal flag
        "reasons",  # Internal reasoning
    }

    output = {k: v for k, v in normalized.items() if k not in internal_fields}

    # Ensure key fields are present
    output.setdefault("severity", decision.severity)
    output.setdefault("trajectory", decision.trajectory)
    output.setdefault("degradation_stage", decision.degradation_stage)
    output.setdefault("suppress", decision.suppress)
    output.setdefault("summary", decision.summary)

    return output


def normalize_finding(finding_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Finding dict."""
    if "confidence" in finding_dict:
        finding_dict["confidence"] = normalize_confidence(finding_dict["confidence"])
    if "magnitude" in finding_dict:
        finding_dict["magnitude"] = normalize_confidence(finding_dict["magnitude"])
    return finding_dict


def validate_numeric_ranges(decision: Decision) -> tuple[bool, list[str]]:
    """Validate all numeric fields are in valid ranges.

    Args:
        decision: Decision to validate

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Confidence fields [0, 1]
    if not (0.0 <= decision.finding_confidence <= 1.0):
        errors.append(f"finding_confidence out of range: {decision.finding_confidence}")
    if not (0.0 <= decision.action_confidence <= 1.0):
        errors.append(f"action_confidence out of range: {decision.action_confidence}")
    if not (0.0 <= decision.transient_score <= 1.0):
        errors.append(f"transient_score out of range: {decision.transient_score}")

    # Persistence frames >= 0
    if decision.persistence_frames_at_level < 0:
        errors.append(f"persistence_frames_at_level negative: {decision.persistence_frames_at_level}")

    # Temporal confidence delta in reasonable range [-1, 1]
    if not (-1.0 <= decision.temporal_confidence_delta <= 1.0):
        errors.append(f"temporal_confidence_delta out of range: {decision.temporal_confidence_delta}")

    return len(errors) == 0, errors
