from __future__ import annotations

from typing import Any

from ui.reasoning import generate_reasoned_response


def render_operational_reasoning_panel(operator_question: str, reasoning_context: dict[str, Any]) -> dict[str, Any]:
    response = generate_reasoned_response(operator_question=operator_question, reasoning_context=reasoning_context)
    gate_decision = reasoning_context.get("gate_decision") or {}
    decision = gate_decision.get("decision") or "SUPPRESS"
    observed_facts = reasoning_context.get("observed_facts")
    if observed_facts is None:
        observed_facts = (response.get("sections") or {}).get("Observed Facts") or []

    temporal_facts = reasoning_context.get("temporal_facts")
    if temporal_facts is None:
        temporal_facts = observed_facts
    transition = reasoning_context.get("transition") or {}
    transition_type = transition.get("type") or "STABLE"

    return {
        "component": "operational_reasoning_panel",
        "title": "Operational Reasoning",
        "question_label": "Decision Query",
        "operator_question": operator_question,
        "observed_facts": temporal_facts,
        "temporal_framing": {
            "observed_facts": temporal_facts,
            "inference": f"System moving away from baseline regime ({transition_type}).",
            "gate_outcome": decision,
        },
        "inference": response,
        "gate_outcome": decision,
        "operational_implication": reasoning_context.get("operational_implication")
        or "Observation only; no doctrine-authorized action.",
        "gate_reference": {
            "decision": decision,
            "reason": gate_decision.get("reason"),
            "doctrine_version": gate_decision.get("doctrine_version"),
            "confidence": gate_decision.get("confidence_label"),
        },
    }
