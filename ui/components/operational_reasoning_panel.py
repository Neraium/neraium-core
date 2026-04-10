from __future__ import annotations

from typing import Any

from ui.reasoning import generate_reasoned_response


def render_operational_reasoning_panel(operator_question: str, reasoning_context: dict[str, Any]) -> dict[str, Any]:
    response = generate_reasoned_response(operator_question=operator_question, reasoning_context=reasoning_context)
    gate_decision = reasoning_context.get("gate_decision") or {}
    return {
        "overlay": "operational_reasoning_panel",
        "title": "Operational Reasoning",
        "subtitle": "Evidence-Bound Reasoning",
        "question_label": "Ask About Current System State",
        "operator_question": operator_question,
        "gate_decision": {
            "decision": gate_decision.get("decision"),
            "reason": gate_decision.get("reason"),
            "doctrine_version": gate_decision.get("doctrine_version"),
        },
        "grounded_answer": response,
    }
