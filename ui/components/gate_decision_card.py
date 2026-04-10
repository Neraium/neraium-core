from __future__ import annotations

from typing import Any


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


def _confidence_label(raw_confidence: str | None) -> str:
    normalized = (raw_confidence or "LOW").upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
        return normalized
    return "LOW"


def render_gate_decision_card(decision: dict[str, Any] | None) -> dict[str, Any]:
    gate = decision or {}
    authority_level = _authority_level(gate.get("decision"))
    refusal_reason = gate.get("refusal_reason")
    reason = gate.get("reason") or gate.get("explanation") or refusal_reason

    return {
        "component": "gate_authority_banner",
        "authority_surface": "Aletheia's Gate",
        "authority_level": authority_level,
        "label": _decision_label(authority_level),
        "confidence": _confidence_label(gate.get("confidence_label")),
        "doctrine_version": gate.get("doctrine_version") or "unknown",
        "timestamp": gate.get("timestamp"),
        "reason": reason,
        "refusal_reason": refusal_reason,
        "criteria_summary": gate.get("criteria_summary") or gate.get("observed_facts") or [],
        "no_signal_admitted": authority_level == "SUPPRESSED",
        "absence_statement": (
            "No admissible system change detected."
            if authority_level == "SUPPRESSED"
            else None
        ),
    }
