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
    prior_severity: str | None = None,
    persistence_frames: int = 0,
    persistence_trajectory: float = 0.0,
) -> SeverityLevel:
    """Map observable metrics to severity level with hysteresis.

    HIGH: Immediate threat (state=ALERT + high drift OR sustained multi-indicator degradation)
    ELEVATED: Significant degradation requiring action
    MODERATE: Changes detected, monitoring required
    LOW: Minor variations, normal operation

    Hysteresis: consider prior severity and persistence to avoid flip-flopping.
    Trajectory: prefer lower classification if system is improving.
    """
    # Base classification from current metrics
    candidate_severity = _base_severity_classification(
        state=state,
        drift_score=drift_score,
        relational_instability=relational_instability,
        system_phase=system_phase,
    )

    # Apply hysteresis: smooth transitions
    if prior_severity is not None:
        candidate_severity = _apply_hysteresis(
            candidate=candidate_severity,
            prior=prior_severity,
            persistence_frames=persistence_frames,
            trajectory=persistence_trajectory,
        )

    return candidate_severity


def _base_severity_classification(
    *,
    state: str,
    drift_score: float,
    relational_instability: float,
    system_phase: str,
) -> SeverityLevel:
    """Base severity without hysteresis."""
    # ALERT state with significant drift: HIGH
    if state == "ALERT" and drift_score > 0.7:
        return "HIGH"

    # ALERT state alone: ELEVATED (not immediate HIGH, requires sustained evidence)
    if state == "ALERT":
        return "ELEVATED"

    # Degrading phase with relational breakdown: ELEVATED
    if system_phase == "degrading":
        if relational_instability > 0.4:
            return "ELEVATED"
        return "ELEVATED" if drift_score > 0.5 else "MODERATE"

    # WATCH state: depends on drift magnitude
    if state == "WATCH":
        if drift_score > 0.6:
            return "ELEVATED"
        if drift_score > 0.4:
            return "MODERATE"
        return "MODERATE"

    # Baseline classification
    if drift_score > 0.6:
        return "MODERATE"

    if relational_instability > 0.5:
        return "MODERATE"

    return "LOW"


def _apply_hysteresis(
    *,
    candidate: SeverityLevel,
    prior: SeverityLevel,
    persistence_frames: int,
    trajectory: float,
) -> SeverityLevel:
    """Apply hysteresis to prevent flip-flopping between severity levels.

    Requires sustained evidence before escalating.
    Prefers lower classifications if trajectory is improving.
    """
    # Improving trajectory: prefer lower classification
    if trajectory < -0.15 and candidate == "ELEVATED":
        return "MODERATE"
    if trajectory < -0.15 and candidate == "HIGH":
        return "ELEVATED"

    # Escalation requirements (require multiple frames at prior level)
    if candidate == "HIGH" and prior in {"MODERATE", "LOW"}:
        # Jumping from MODERATE/LOW to HIGH: requires sustained evidence
        if persistence_frames < 2:
            return prior  # Wait for more evidence

    if candidate == "ELEVATED" and prior == "MODERATE":
        # MODERATE → ELEVATED: requires at least 2 frames of elevation signals
        if persistence_frames < 2:
            return "MODERATE"

    # Never downgrade from HIGH immediately (requires clear recovery)
    if prior == "HIGH" and candidate in {"ELEVATED", "MODERATE", "LOW"}:
        # Allow downgrade only if stable for 2+ frames
        if persistence_frames < 2:
            return "HIGH"

    return candidate


def compute_suppress_flag(
    *,
    severity: SeverityLevel,
    transient_score: float,
    finding_confidence: float,
    persistence_frames: int = 0,
    is_first_appearance: bool = False,
    frame_number: int = 0,
) -> bool:
    """Determine if we should suppress this finding from operators.

    Never suppresses CRITICAL/HIGH severity.
    Requires sustained evidence before surfacing ELEVATED/MODERATE.
    Applies startup suppression to reduce false positives.
    """
    # CRITICAL/HIGH never suppressed
    if severity == "HIGH":
        return False

    # High confidence findings almost never suppressed
    if finding_confidence > 0.85:
        return False

    # Startup suppression (first 8 frames)
    startup_window = 8
    if frame_number < startup_window:
        # First appearance at ELEVATED: suppress until confirmed (2+ frames)
        if is_first_appearance and severity in {"ELEVATED", "MODERATE"}:
            if persistence_frames < 2:
                return True
        # High transience score during startup
        if transient_score > 0.65 and severity == "MODERATE":
            if persistence_frames < 2:
                return True

    # Transience suppression
    if transient_score > 0.75:
        if severity in {"MODERATE", "LOW"}:
            return True

    if transient_score > 0.65 and severity == "MODERATE" and finding_confidence < 0.6:
        return True

    # Low confidence suppression
    if severity == "MODERATE":
        if finding_confidence < 0.4 and transient_score > 0.7:
            return True
        if finding_confidence < 0.3:
            return True

    if severity == "LOW":
        if finding_confidence < 0.5 and transient_score > 0.4:
            return True
        if finding_confidence < 0.3:
            return True

    return False


def compute_summary(
    *,
    severity: SeverityLevel,
    state: str,
    primary_finding: str,
    suppress: bool,
    persistence_frames: int = 0,
) -> str:
    """Generate a one-line summary of the decision."""
    persistence_hint = f" ({persistence_frames}f)" if persistence_frames > 1 else ""

    if suppress:
        return f"[SUPPRESSED] {primary_finding} (transient/low confidence)"

    if severity == "HIGH":
        return f"⚠️ HIGH{persistence_hint}: {primary_finding} — Immediate attention required"

    if severity == "ELEVATED":
        return f"⚡ ELEVATED{persistence_hint}: {primary_finding} — Action needed soon"

    if severity == "MODERATE":
        return f"📊 MODERATE{persistence_hint}: {primary_finding} — Monitor closely"

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
