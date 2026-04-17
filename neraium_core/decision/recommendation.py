"""Generate human-facing recommendations.

Advisory only; no control authority or actuation.
"""

from __future__ import annotations

from typing import Any
from neraium_core.decision.models import Recommendation, SeverityLevel


def recommend_action(
    *,
    severity: SeverityLevel,
    state: str,
    drift_score: float,
    time_to_instability: float | None = None,
    top_signals: list[str] | None = None,
    action_confidence: float = 0.5,
    persistence_frames: int = 0,
    last_action: str | None = None,
) -> Recommendation | None:
    """Generate a recommendation based on findings with escalation.

    Returns None if no action needed.
    Implements escalation logic: monitor → inspect → escalate → urgent
    All recommendations are "monitor/inspect/schedule" — no control directives.
    """
    if not top_signals:
        top_signals = []

    target = top_signals[0] if top_signals else None

    if severity == "HIGH":
        if time_to_instability and time_to_instability < 1.0:
            return Recommendation(
                action="urgent_inspection_required",
                target=target,
                urgency="HIGH",
                rationale="System approaching critical threshold; immediate operator attention needed.",
                time_window_hours=0.5,
                confidence=action_confidence,
            )
        else:
            return Recommendation(
                action="escalate_to_operations",
                target=target,
                urgency="HIGH",
                rationale="Structural instability detected across multiple signals.",
                time_window_hours=1.0,
                confidence=action_confidence,
            )

    if severity == "ELEVATED":
        # Escalation logic: if sustained at ELEVATED, escalate to inspection
        if persistence_frames >= 4 and last_action == "increase_monitoring_cadence":
            return Recommendation(
                action="schedule_inspection",
                target=target,
                urgency="ELEVATED",
                rationale=f"System sustained at elevated severity. Escalating to inspection of {target if target else 'key systems'}.",
                time_window_hours=2.0,
                confidence=action_confidence,
            )

        if drift_score > 0.7:
            return Recommendation(
                action="schedule_inspection",
                target=target,
                urgency="ELEVATED",
                rationale=f"Major structural misalignment detected. Inspect {target if target else 'key systems'} within 4 hours.",
                time_window_hours=4.0,
                confidence=action_confidence,
            )
        else:
            return Recommendation(
                action="increase_monitoring_cadence",
                target=target,
                urgency="ELEVATED",
                rationale="System trending toward instability. Increase sampling frequency to catch changes faster.",
                time_window_hours=0.5,
                confidence=action_confidence,
            )

    if severity == "MODERATE":
        if state in {"WATCH"}:
            return Recommendation(
                action="schedule_maintenance",
                target=target,
                urgency="MODERATE",
                rationale="System showing signs of wear. Schedule maintenance within 24 hours.",
                time_window_hours=24.0,
                confidence=action_confidence,
            )
        else:
            return Recommendation(
                action="monitor_closely",
                target=target,
                urgency="MODERATE",
                rationale="Changes detected but within safe operating envelope. Continue normal monitoring.",
                time_window_hours=None,
                confidence=action_confidence,
            )

    if severity == "LOW":
        return Recommendation(
            action="monitor",
            target=target,
            urgency="LOW",
            rationale="Minor variations detected. No immediate action needed; standard monitoring sufficient.",
            time_window_hours=None,
            confidence=action_confidence,
        )

    return None


def should_emit_recommendation(
    *,
    current_action: str,
    last_action: str | None,
    frames_since_last: int,
    severity_changed: bool,
) -> bool:
    """Determine if we should emit this recommendation to avoid repetition.

    Avoids repeating identical recommendations every frame.
    Re-emits only if: action changed, severity escalated, or enough frames passed.
    """
    # First recommendation: always emit
    if last_action is None:
        return True

    # Different action: always emit
    if current_action != last_action:
        return True

    # Severity escalated: re-emit with new context
    if severity_changed:
        return True

    # Re-emit every 5 frames to keep operator informed
    if frames_since_last >= 5:
        return True

    return False


def recommendation_hash(action: str, target: str | None, urgency: str) -> str:
    """Create stable hash for deduplication."""
    return f"{action}:{target or 'none'}:{urgency}"


def format_recommendation(rec: Recommendation | None) -> str:
    """Format a recommendation as human-readable text."""
    if not rec:
        return "No action recommended at this time."

    target_text = f" ({rec.target})" if rec.target else ""
    time_text = f" within {rec.time_window_hours:.1f} hours" if rec.time_window_hours else ""

    return f"{rec.action.replace('_', ' ').title()}{target_text}{time_text}: {rec.rationale}"
