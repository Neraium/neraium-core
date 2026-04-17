"""Decision policies: rules for when to surface, suppress, or recommend."""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import SeverityLevel


def classify_severity(
    *,
    state: str,
    drift_score: float,
    relational_instability: float,
    system_phase: str,
) -> SeverityLevel:
    """Map observable metrics to severity level.

    Aligned with existing system: HIGH | ELEVATED | MODERATE | LOW

    HIGH: Immediate threat to safe operation (state=ALERT + high drift)
    ELEVATED: Significant degradation, needs action soon
    MODERATE: Changes detected, needs monitoring
    LOW: Minor variations, normal operation
    """
    if state == "ALERT" and drift_score > 0.7:
        return "HIGH"

    if state == "ALERT":
        return "ELEVATED"

    if state == "WATCH":
        if drift_score > 0.5:
            return "ELEVATED"
        return "MODERATE"

    if system_phase == "degrading":
        if relational_instability > 0.5:
            return "ELEVATED"
        return "MODERATE"

    if drift_score > 0.6:
        return "MODERATE"

    if relational_instability > 0.5:
        return "MODERATE"

    return "LOW"


def compute_suppress_flag(
    *,
    severity: SeverityLevel,
    transient_score: float,
    finding_confidence: float,
) -> bool:
    """Determine if we should suppress this finding from operators.

    Never suppresses HIGH severity.
    HIGH-confidence or ELEVATED findings are not suppressed.
    Low findings with low confidence + high transience can be suppressed.
    """
    if severity == "HIGH":
        return False

    if finding_confidence > 0.8:
        return False

    if severity in {"ELEVATED"}:
        return False

    if severity == "MODERATE" and finding_confidence < 0.4:
        if transient_score > 0.7:
            return True

    if severity == "LOW":
        if finding_confidence < 0.5 and transient_score > 0.6:
            return True

    return False


def compute_summary(
    *,
    severity: SeverityLevel,
    state: str,
    primary_finding: str,
    suppress: bool,
) -> str:
    """Generate a one-line summary of the decision."""
    if suppress:
        return f"[SUPPRESSED] {primary_finding} (transient/low confidence)"

    if severity == "HIGH":
        return f"⚠️ HIGH: {primary_finding} — Immediate attention required"

    if severity == "ELEVATED":
        return f"⚡ ELEVATED: {primary_finding} — Action needed soon"

    if severity == "MODERATE":
        return f"📊 MODERATE: {primary_finding} — Monitor closely"

    return f"ℹ️ {primary_finding}"


def should_recommend(
    *,
    severity: SeverityLevel,
    action_confidence: float,
) -> bool:
    """Determine if we should make a recommendation to the operator."""
    if action_confidence < 0.4:
        return False

    if severity == "HIGH":
        return True

    if severity == "ELEVATED":
        return action_confidence > 0.5

    if severity == "MODERATE":
        return action_confidence > 0.6

    return False
