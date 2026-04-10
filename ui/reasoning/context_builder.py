from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ui.reasoning.models import AdmittedEvent
from ui.core_integration import SystemState


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_event_admitted(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return None


def _summarize_event(event: AdmittedEvent, *, drift_threshold: float, stability_threshold: float) -> str:
    reasons: list[str] = []
    if event.drift >= drift_threshold:
        reasons.append(f"drift {event.drift:.3f} exceeded {drift_threshold:.3f}")
    if event.stability <= stability_threshold:
        reasons.append(f"stability {event.stability:.3f} stayed below {stability_threshold:.3f}")
    if not reasons:
        reasons.append("no admission threshold breach was observed")
    return "; ".join(reasons)


def build_admitted_events(
    records: list[dict[str, Any]] | None,
    *,
    drift_threshold: float = 0.55,
    stability_threshold: float = 0.45,
) -> list[AdmittedEvent]:
    events: list[AdmittedEvent] = []
    for row in records or []:
        drift = _as_float(row.get("structural_drift_score", row.get("drift", 0.0)), 0.0)
        stability = _as_float(row.get("relational_stability_score", row.get("stability", 1.0)), 1.0)
        admitted_from_row = _parse_event_admitted(row.get("event_admitted"))
        event_admitted = admitted_from_row if admitted_from_row is not None else bool(
            drift >= drift_threshold and stability <= stability_threshold
        )
        event = AdmittedEvent(
            timestamp=str(row.get("timestamp") or "unknown"),
            drift=round(drift, 6),
            stability=round(stability, 6),
            regime=str(row.get("regime_name") or row.get("regime") or "unknown"),
            event_admitted=event_admitted,
            evidence_summary=str(
                row.get("evidence_summary")
                or _summarize_event(
                    AdmittedEvent(
                        timestamp="",
                        drift=drift,
                        stability=stability,
                        regime="",
                        event_admitted=event_admitted,
                        evidence_summary="",
                    ),
                    drift_threshold=drift_threshold,
                    stability_threshold=stability_threshold,
                )
            ),
        )
        events.append(event)
    return events


def build_reasoning_context(
    state: SystemState,
    records: list[dict[str, Any]] | None = None,
    *,
    gate_decision: dict[str, Any] | None = None,
    max_recent_events: int = 6,
) -> dict[str, Any]:
    admitted_events = [event for event in build_admitted_events(records) if event.event_admitted]
    transition_point = admitted_events[0].timestamp if admitted_events else None

    top_signals = None
    if records:
        latest = records[-1]
        candidate = latest.get("top_contributing_signals") or latest.get("signal_attribution")
        if isinstance(candidate, list):
            top_signals = candidate[:5]

    drift_direction = "elevated"
    if state.velocity[0] < 0:
        drift_direction = "easing"

    stability_direction = "low"
    if state.velocity[1] < 0:
        stability_direction = "recovering"

    gate = gate_decision or {}
    context = {
        "current_state": {
            "timestamp": state.position.t,
            "regime": state.regime_state,
            "system_health": state.system_health,
            "confidence": round(state.confidence, 4),
            "drift": round(state.drift_intensity, 4),
            "stability": round(1.0 - state.position.y, 4),
            "velocity": {"dx": round(state.velocity[0], 6), "dy": round(state.velocity[1], 6)},
        },
        "gate_decision": {
            "decision": gate.get("decision"),
            "reason": gate.get("refusal_reason") or gate.get("explanation") or gate.get("reason"),
            "doctrine_version": gate.get("doctrine_version"),
            "confidence_label": gate.get("confidence_label"),
        },
        "recent_admitted_events": [asdict(event) for event in admitted_events[-max_recent_events:]],
        "transition_point": transition_point,
        "drift_summary": f"Structural drift is currently {drift_direction}.",
        "stability_summary": f"Relational stability remains {stability_direction}.",
        "top_contributing_signals": top_signals,
        "chart_replay_summary": {
            "trajectory_points": len(state.trajectory_history),
            "projected_points": len(state.projected_cone),
            "current_position": {"x": state.position.x, "y": state.position.y},
        },
    }
    return context
