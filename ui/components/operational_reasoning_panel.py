from __future__ import annotations

from typing import Any

from ui.reasoning import generate_reasoned_response


def render_operational_reasoning_panel(operator_question: str, reasoning_context: dict[str, Any]) -> dict[str, Any]:
    response = generate_reasoned_response(question=operator_question, context=reasoning_context)
    return {
        "overlay": "operational_reasoning_panel",
        "title": "Operational Reasoning",
        "subtitle": "Evidence-Bound Reasoning",
        "question_label": "Ask About Current System State",
        "operator_question": operator_question,
        "grounded_answer": response,
    }
