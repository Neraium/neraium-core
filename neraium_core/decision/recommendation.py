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
) -> Recommendation | None:
    """Generate a recommendation based on findings.

    Returns None if no action needed.
    All recommendations are "monitor/inspect/schedule" — no control directives.
    """
    if not top_signals:
        top_signals = []

    if severity == "CRITICAL":
        if time_to_instability and time_to_instability < 1.0:
            return Recommendation(
                action="urgent_inspection_required",
                target=top_signals[0] if top_signals else None,
                urgency="CRITICAL",
                rationale="System approaching critical threshold; immediate operator attention needed.",
                time_window_hours=0.5,
                confidence=action_confidence,
            )
        else:
            return Recommendation(
                action="escalate_to_operations",
                target=top_signals[0] if top_signals else None,
                urgency="CRITICAL",
                rationale="Critical structural instability detected across multiple signals.",
                time_window_hours=1.0,
                confidence=action_confidence,
            )

    if severity == "HIGH":
        if drift_score > 0.7:
            return Recommendation(
                action="schedule_inspection",
                target=top_signals[0] if top_signals else None,
                urgency="HIGH",
                rationale=f"Major structural misalignment detected. Inspect {top_signals[0] if top_signals else 'key systems'} within 4 hours.",
                time_window_hours=4.0,
                confidence=action_confidence,
            )
        else:
            return Recommendation(
                action="increase_monitoring_cadence",
                target=top_signals[0] if top_signals else None,
                urgency="HIGH",
                rationale="System trending toward instability. Increase sampling frequency to catch changes faster.",
                time_window_hours=0.5,
                confidence=action_confidence,
            )

    if severity == "MODERATE":
        if state in {"WATCH"}:
            return Recommendation(
                action="schedule_maintenance",
                target=top_signals[0] if top_signals else None,
                urgency="MODERATE",
                rationale="System showing signs of wear. Schedule maintenance within 24 hours.",
                time_window_hours=24.0,
                confidence=action_confidence,
            )
        else:
            return Recommendation(
                action="monitor_closely",
                target=top_signals[0] if top_signals else None,
                urgency="MODERATE",
                rationale="Changes detected but within safe operating envelope. Continue normal monitoring.",
                time_window_hours=None,
                confidence=action_confidence,
            )

    if severity == "LOW":
        return Recommendation(
            action="monitor",
            target=top_signals[0] if top_signals else None,
            urgency="LOW",
            rationale="Minor variations detected. No immediate action needed; standard monitoring sufficient.",
            time_window_hours=None,
            confidence=action_confidence,
        )

    return None


def format_recommendation(rec: Recommendation | None) -> str:
    """Format a recommendation as human-readable text."""
    if not rec:
        return "No action recommended at this time."

    target_text = f" ({rec.target})" if rec.target else ""
    time_text = f" within {rec.time_window_hours:.1f} hours" if rec.time_window_hours else ""

    return f"{rec.action.replace('_', ' ').title()}{target_text}{time_text}: {rec.rationale}"
