from __future__ import annotations

from typing import Any


def render_gate_decision_card(decision: dict[str, Any] | None) -> dict[str, Any]:
    gate = decision or {}
    return {
        "component": "gate_decision",
        "title": "Aletheia's Gate",
        "decision": gate.get("decision"),
        "confidence": gate.get("confidence_label"),
        "reason": gate.get("refusal_reason") or gate.get("explanation"),
        "doctrine_version": gate.get("doctrine_version"),
        "criteria": gate.get("criteria_results"),
        "timestamp": gate.get("timestamp"),
    }
