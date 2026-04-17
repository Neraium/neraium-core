"""Transient event detection and gating.

Identifies temporary spikes and common safe transitions that shouldn't trigger alerts.
"""

from __future__ import annotations

from typing import Any


def score_transient_likelihood(
    *,
    drift_score: float,
    drift_trend: float,
    shock_activity: float,
    system_phase: str,
    state: str,
    drift_history: list[float] | None = None,
) -> float:
    """Score likelihood this is a transient event that will self-resolve.

    Returns [0, 1] where 1 = almost certainly transient.
    """
    score = 0.0

    if shock_activity > 0.75:
        score += 0.65
    elif shock_activity > 0.6:
        score += 0.45

    if state in {"STABLE", "NOMINAL"}:
        score += 0.15

    if drift_score < 0.3:
        score += 0.1

    if drift_trend < 0.0:
        score += 0.15
    elif drift_trend > 0.15:
        score -= 0.1

    if system_phase in {"stable", "recovering"}:
        score += 0.15
    elif system_phase in {"transitional"}:
        score += 0.1

    if drift_history and len(drift_history) >= 3:
        recent = drift_history[-3:]
        variance = _compute_variance(recent)
        if variance > 0.08:
            score -= 0.2
        elif variance < 0.02:
            score += 0.05

    return min(1.0, max(0.0, score))


def is_known_safe_transient(
    *,
    state: str,
    system_phase: str,
    regime_name: str | None = None,
    recent_events: list[str] | None = None,
) -> bool:
    """Check if this looks like a known safe transient event.

    Examples:
    - Startup spikes (first N frames)
    - Scheduled maintenance windows
    - Oscillations after mode changes
    """
    if regime_name in {"startup", "initialization", "calibration"}:
        return True

    if system_phase == "recovering":
        return True

    if recent_events is None:
        recent_events = []

    known_safe = {
        "startup_initialization",
        "calibration_in_progress",
        "scheduled_maintenance",
        "mode_transition_complete",
    }

    if any(event in known_safe for event in recent_events):
        return True

    return False


def apply_transient_suppression(
    *,
    severity: str,
    transient_score: float,
    suppress_currently: bool,
) -> tuple[bool, str]:
    """Determine if we should suppress based on transient likelihood.

    Returns (suppress, reason).
    Never suppresses HIGH severity.
    """
    if severity == "HIGH":
        return False, ""

    if transient_score > 0.75:
        if severity in {"LOW", "MODERATE"}:
            return True, f"likely transient (score {transient_score:.2f})"

    if transient_score > 0.6 and severity == "LOW":
        return True, f"possibly transient (score {transient_score:.2f})"

    return suppress_currently, ""


def _compute_variance(values: list[float]) -> float:
    """Simple variance calculation."""
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
