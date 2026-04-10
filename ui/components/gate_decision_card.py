from __future__ import annotations

from typing import Any


def _reality_status(decision: str | None) -> str:
    normalized = (decision or "SUPPRESS").upper()
    if normalized == "ADMIT":
        return "Change is real"
    if normalized in {"ADMISSIBILITY_VOID", "VOID"}:
        return "Signal not valid"
    return "No confirmed change"


def _authority_level(decision: str | None) -> str:
    normalized = (decision or "SUPPRESS").upper()
    if normalized == "ADMIT":
        return "ADMITTED"
    if normalized in {"ADMISSIBILITY_VOID", "VOID"}:
        return "VOID"
    return "SUPPRESSED"


def _decision_label(authority_level: str) -> str:
    if authority_level == "ADMITTED":
        return "ADMITTED AS REAL: COHERENT SYSTEM REORGANIZATION DETECTED"
    if authority_level == "VOID":
        return "ADMISSIBILITY VOID: EVIDENCE INSUFFICIENT OR INCOHERENT"
    return "NO ACTIONABLE SIGNAL ADMITTED"


def _authority_statement(authority_level: str) -> str:
    if authority_level == "ADMITTED":
        return "System reorganization detected and admitted."
    if authority_level == "VOID":
        return "Signal invalidated due to incoherent evidence."
    return "No admissible structural change detected."


def _confidence_label(raw_confidence: str | None) -> str:
    normalized = (raw_confidence or "LOW").upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized
    return "LOW"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _transition_intensity(transition: dict[str, Any], persistence_minutes: Any) -> str:
    drift = abs(_to_float(transition.get("delta_drift")) or 0.0)
    stability_drop = max(0.0, -(_to_float(transition.get("delta_stability")) or 0.0))
    persistence = max(0.0, _to_float(persistence_minutes) or 0.0)

    if drift >= 0.30 or stability_drop >= 0.30 or persistence >= 60:
        return "HIGH"
    if drift >= 0.12 or stability_drop >= 0.12 or persistence >= 20:
        return "MODERATE"
    return "LOW"


def _normalize_transition_type(transition_type: Any) -> str:
    if isinstance(transition_type, str):
        normalized = transition_type.strip().upper()
        if normalized:
            return normalized
    return "STABLE"


def _operator_takeaway(authority_level: str, transition_type: str, transition_intensity: str) -> str:
    if authority_level == "ADMITTED":
        return (
            f"System has entered a {transition_intensity.lower()}-intensity "
            f"{transition_type.lower()} transition."
        )
    if authority_level == "VOID":
        return "Observed signal remains invalid for determining a persistent transition."
    return "No persistent deviation from baseline has been confirmed."


def _risk_direction(transition_type: str, transition: dict[str, Any], authority_level: str) -> str:
    if authority_level == "VOID":
        return "UNCERTAIN"

    drift = _to_float(transition.get("delta_drift")) or 0.0
    stability_delta = _to_float(transition.get("delta_stability")) or 0.0
    coherence_delta = _to_float(transition.get("delta_coherence")) or 0.0
    normalized_transition = (transition_type or "STABLE").upper()

    degrading_transition = normalized_transition in {"INSTABILITY", "DEGRADING", "DIVERGENCE"}
    stable_transition = normalized_transition in {"STABLE", "RECOVERY"}
    no_directional_evidence = drift == 0.0 and stability_delta == 0.0 and coherence_delta == 0.0

    if degrading_transition or drift >= 0.12 or stability_delta <= -0.12:
        return "DEGRADING"
    if authority_level == "ADMITTED" and no_directional_evidence:
        return "UNCERTAIN"
    if stable_transition and abs(drift) < 0.12 and abs(stability_delta) < 0.12:
        return "STABLE"
    return "UNCERTAIN"


def _trajectory_statement(authority_level: str, risk_direction: str) -> str:
    if authority_level == "VOID":
        return "Signal coherence insufficient to determine system direction."
    if authority_level == "SUPPRESSED" and risk_direction == "STABLE":
        return "No sustained directional change detected."
    if risk_direction == "DEGRADING":
        return "System is progressing away from stable operating conditions."
    if risk_direction == "STABLE":
        return "System direction remains within stable operating conditions."
    return "Directional trajectory remains uncertain under current evidence."


def _if_sustained_statement(risk_direction: str, authority_level: str) -> str:
    if authority_level == "VOID" or risk_direction == "UNCERTAIN":
        return "If sustained, this condition indicates: Insufficient evidence to project system evolution."
    if risk_direction == "DEGRADING":
        return "If sustained, this condition indicates: Potential transition into a new operating regime."
    return "If sustained, this condition indicates: No expected change in system behavior."


def render_gate_decision_card(decision: dict[str, Any] | None) -> dict[str, Any]:
    gate = decision or {}
    authority_level = _authority_level(gate.get("decision"))
    refusal_reason = gate.get("refusal_reason")
    reason = gate.get("reason") or gate.get("explanation") or refusal_reason

    transition = gate.get("transition") if isinstance(gate.get("transition"), dict) else {}
    transition_type = _normalize_transition_type(transition.get("type"))
    intensity = _transition_intensity(transition, gate.get("persistence_minutes"))
    risk_direction = _risk_direction(transition_type, transition, authority_level)
    timestamp = gate.get("timestamp")

    return {
        "component": "gate_authority_banner",
        "visual_priority": "primary",
        "layout_weight": "dominant",
        "reality_status": _reality_status(gate.get("decision")),
        "authority_surface": "Aletheia's Gate",
        "authority_level": authority_level,
        "label": _decision_label(authority_level),
        "authority_statement": _authority_statement(authority_level),
        "confidence": _confidence_label(gate.get("confidence_label")),
        "transition_type": transition_type,
        "transition_intensity": intensity,
        "operator_takeaway_label": "Operator Takeaway",
        "operator_takeaway": _operator_takeaway(authority_level, transition_type, intensity),
        "trajectory_statement": _trajectory_statement(authority_level, risk_direction),
        "risk_direction": risk_direction,
        "if_sustained_statement": _if_sustained_statement(risk_direction, authority_level),
        "doctrine_version": gate.get("doctrine_version") or "unknown",
        "timestamp": timestamp,
        "timestamp_label": "Change evaluated at",
        "timestamp_display": f"Change evaluated at: {timestamp}" if timestamp else "Change evaluated at: unknown",
        "reason": reason,
        "refusal_reason": refusal_reason,
        "criteria_summary": gate.get("criteria_summary") or gate.get("observed_facts") or [],
        "what_changed": {
            "title": "What Changed",
            "structural_drift": transition.get("delta_drift"),
            "stability": transition.get("delta_stability"),
            "coherence": transition.get("delta_coherence"),
            "persistence_minutes": gate.get("persistence_minutes"),
            "transition_type": transition_type,
            "transition_intensity": intensity,
        },
        "no_signal_admitted": authority_level == "SUPPRESSED",
        "absence_statement": (
            "No admissible system change detected."
            if authority_level == "SUPPRESSED"
            else None
        ),
    }
