from __future__ import annotations

from typing import Any


def evaluate_signal(timeseries: list[dict[str, Any]], unit_summary: dict[str, Any]) -> dict[str, Any]:
    """Evaluate operator-facing signal from raw SII outputs.

    This layer is intentionally simple and rule-based. It does not alter SII
    metrics; it only interprets already-computed drift/instability/phase values.
    """

    ordered = sorted(timeseries, key=lambda row: int(row.get("cycle", 0)))
    if not ordered:
        return {
            "signal_emitted": False,
            "signal_strength": "low",
            "confidence": "low",
            "reason": ["no data available"],
            "phase": "stable",
            "risk_level": "LOW",
        }

    instability = [float(row.get("composite_instability", 0.0)) for row in ordered]
    drift = [float(row.get("structural_drift_score", 0.0)) for row in ordered]
    phases = [str(row.get("phase", "stable")) for row in ordered]

    latest_phase = phases[-1]
    latest_risk = str(ordered[-1].get("risk_level", "LOW"))
    latest_instability = instability[-1]

    reasons: list[str] = []
    suppress_reasons: list[str] = []

    sustained_instability = _is_sustained_instability(instability)
    drift_increasing = _is_increasing_recent(drift)
    phase_consistent = _has_consistent_phase_progression(phases)

    if sustained_instability:
        reasons.append("composite instability is sustained above threshold")
    if drift_increasing:
        reasons.append("structural drift is increasing over recent cycles")
    if phase_consistent:
        reasons.append("phase progression is consistent")

    if _has_spike_then_drop(instability):
        suppress_reasons.append("instability spike dropped too quickly")
    if _has_phase_oscillation(phases):
        suppress_reasons.append("phase is oscillating")

    emit_signal = (
        latest_instability > 0.7
        and sustained_instability
        and drift_increasing
        and phase_consistent
        and not suppress_reasons
    )

    if suppress_reasons:
        reasons.extend(f"suppressed: {reason}" for reason in suppress_reasons)

    if emit_signal:
        signal_strength = _signal_strength(latest_instability, unit_summary)
        confidence = _confidence_level(sustained_instability, drift_increasing, phase_consistent)
    else:
        signal_strength = "low"
        confidence = "low" if suppress_reasons else "medium"

    return {
        "signal_emitted": emit_signal,
        "signal_strength": signal_strength,
        "confidence": confidence,
        "reason": reasons or ["insufficient evidence for a stable signal"],
        "phase": latest_phase,
        "risk_level": latest_risk,
    }


def _is_sustained_instability(values: list[float], window: int = 4) -> bool:
    if len(values) < window:
        return False
    tail = values[-window:]
    return sum(1 for value in tail if value > 0.7) >= window - 1 and tail[-1] >= tail[0]


def _is_increasing_recent(values: list[float], window: int = 5) -> bool:
    if len(values) < window:
        return False
    recent = values[-window:]
    return recent[-1] > recent[0] and sum(1 for i in range(1, len(recent)) if recent[i] >= recent[i - 1]) >= 3


def _has_consistent_phase_progression(phases: list[str]) -> bool:
    rank = {"stable": 0, "drift": 1, "unstable": 2}
    ranked = [rank.get(phase, 0) for phase in phases]
    regressions = sum(1 for i in range(1, len(ranked)) if ranked[i] < ranked[i - 1])
    return regressions <= 1


def _has_spike_then_drop(values: list[float]) -> bool:
    for i in range(1, len(values) - 1):
        if values[i] > 0.8 and values[i + 1] < values[i] - 0.25:
            return True
    return False


def _has_phase_oscillation(phases: list[str]) -> bool:
    for i in range(len(phases) - 2):
        if phases[i] == "unstable" and phases[i + 1] == "stable" and phases[i + 2] == "unstable":
            return True
    return False


def _signal_strength(latest_instability: float, unit_summary: dict[str, Any]) -> str:
    peak = float(unit_summary.get("peak_instability", latest_instability))
    if latest_instability >= 0.9 or peak >= 0.9:
        return "high"
    if latest_instability >= 0.8 or peak >= 0.8:
        return "medium"
    return "low"


def _confidence_level(sustained_instability: bool, drift_increasing: bool, phase_consistent: bool) -> str:
    score = sum([sustained_instability, drift_increasing, phase_consistent])
    if score == 3:
        return "high"
    if score == 2:
        return "medium"
    return "low"
