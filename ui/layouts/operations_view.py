from __future__ import annotations

from typing import Any

from ui.components import (
    render_event_ledger,
    render_gate_decision_card,
    render_operational_reasoning_panel,
    render_structural_flow_viz,
)
from ui.core_integration import SystemState


def _timeline_strip(state: SystemState, gate: dict[str, Any]) -> dict[str, Any]:
    recent = state.timeline[-5:]
    stages = []
    for event in recent:
        stage = "baseline"
        if abs(event.drift_delta) > 0.08:
            stage = "transition"
        stages.append({"t": event.t, "stage": stage})

    decision = (gate.get("decision") or "SUPPRESS").upper()
    admitted_stage = "admitted" if decision == "ADMIT" else ("suppressed" if decision == "SUPPRESS" else "void")
    stages.append({"t": gate.get("timestamp") or state.position.t, "stage": admitted_stage})
    return {"title": "Timeline Strip", "sequence": stages}


def build_operations_view(
    state: SystemState,
    *,
    width_px: int = 1440,
    reasoning_context: dict[str, Any] | None = None,
    operator_question: str = "What is admissible right now?",
    gate_decision: dict[str, Any] | None = None,
) -> dict[str, object]:
    context = dict(reasoning_context or {})
    raw_gate = gate_decision or context.get("gate_decision") or {}
    gate = dict(raw_gate)
    gate["decision"] = gate.get("decision") or "SUPPRESS"

    context_gate = context.get("gate_decision") if isinstance(context.get("gate_decision"), dict) else {}
    if (
        gate_decision is not None
        or "gate_decision" not in context
        or context_gate.get("decision") != gate["decision"]
    ):
        context["gate_decision"] = {
            "decision": gate["decision"],
            "reason": gate.get("refusal_reason") or gate.get("explanation") or gate.get("reason"),
            "doctrine_version": gate.get("doctrine_version"),
            "confidence_label": gate.get("confidence_label"),
            "transition": gate.get("transition"),
            "persistence_minutes": gate.get("persistence_minutes"),
        }

    return {
        "mode": "operations",
        "viewport": "mobile" if width_px < 760 else "desktop",
        "hierarchy": "Reality Status → Gate Authority → Secondary Context",
        "zones": {
            "gate": {
                "layout": "full_width_authority_banner",
                "visual_emphasis": "maximum",
                "brightness": "100%",
                "content": render_gate_decision_card(gate),
            },
            "system_state": {
                "title": "System Context",
                "role": "supporting_context",
                "visual_emphasis": "secondary",
                "brightness": "55%",
                "content": render_structural_flow_viz(state, gate_decision=gate),
                "timeline_strip": _timeline_strip(state, gate),
            },
            "reasoning": {
                "role": "secondary_reasoning",
                "visual_emphasis": "secondary",
                "brightness": "55%",
                "content": render_operational_reasoning_panel(operator_question, context),
            },
            "record": {
                "role": "audit_evidence",
                "visual_emphasis": "secondary",
                "brightness": "55%",
                "content": render_event_ledger(context.get("recent_admitted_events")),
            },
        },
    }
