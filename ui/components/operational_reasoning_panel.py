from __future__ import annotations

from typing import Any

from ui.reasoning import generate_reasoned_response


def render_operational_reasoning_panel(operator_question: str, reasoning_context: dict[str, Any]) -> dict[str, Any]:
    response = generate_reasoned_response(operator_question=operator_question, reasoning_context=reasoning_context)
    gate_decision = reasoning_context.get("gate_decision") or {}
    decision = gate_decision.get("decision") or "SUPPRESS"

    return {
        "component": "operational_reasoning_panel",
        "title": "Doctrine-Bound Reasoning Context",
        "question_label": "Observed System Question",
        "operator_question": operator_question,
        "observed_facts": reasoning_context.get("observed_facts") or [],
        "inference": response,
        "gate_outcome": decision,
        "operational_implication": reasoning_context.get("operational_implication")
        or "Observational implication only; doctrine does not authorize prescriptive action.",
        "gate_reference": {
            "decision": decision,
            "reason": gate_decision.get("reason"),
            "doctrine_version": gate_decision.get("doctrine_version"),
            "confidence": gate_decision.get("confidence_label"),
        },
    }
