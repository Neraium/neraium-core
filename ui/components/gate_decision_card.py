from __future__ import annotations

from typing import Any


def render_gate_decision_card(decision: dict[str, Any] | None) -> dict[str, Any]:
    gate = decision or {}
    return {
        "component": "gate_decision_card",
        "authority": "Aletheia's Gate",
        "decision": gate.get("decision"),
        "doctrine_version": gate.get("doctrine_version"),
        "explanation": gate.get("explanation"),
        "reason": gate.get("explanation"),
        "refusal_reason": gate.get("refusal_reason"),
        "confidence_label": gate.get("confidence_label"),
        "timestamp": gate.get("timestamp"),
        "criteria_results": gate.get("criteria_results") or {},
    }
