from __future__ import annotations

from typing import Any

from ui.components import (
    render_event_ledger,
    render_gate_decision_card,
    render_operational_reasoning_panel,
    render_structural_flow_viz,
)
from ui.core_integration import SystemState


def build_operations_view(
    state: SystemState,
    *,
    width_px: int = 1440,
    reasoning_context: dict[str, Any] | None = None,
    operator_question: str = "What is admissible right now?",
    gate_decision: dict[str, Any] | None = None,
) -> dict[str, object]:
    context = dict(reasoning_context or {})
    gate = gate_decision or {"decision": "SUPPRESS"}

    context["gate_decision"] = {
        "decision": gate.get("decision") or "SUPPRESS",
        "reason": gate.get("refusal_reason") or gate.get("explanation") or gate.get("reason"),
        "doctrine_version": gate.get("doctrine_version"),
        "confidence_label": gate.get("confidence_label"),
    }

    return {
        "mode": "operations",
        "viewport": "mobile" if width_px < 760 else "desktop",
        "hierarchy": "Gate Authority → Supporting Context → Doctrine-Bound Reasoning → Evidence Record",
        "zones": {
            "gate": {
                "layout": "full_width_authority_banner",
                "content": render_gate_decision_card(gate),
            },
            "system_state": {
                "title": "System Context",
                "role": "supporting_context",
                "content": render_structural_flow_viz(state, gate_decision=gate),
            },
            "reasoning": {
                "role": "secondary_reasoning",
                "content": render_operational_reasoning_panel(operator_question, context),
            },
            "record": {
                "role": "audit_evidence",
                "content": render_event_ledger(context.get("recent_admitted_events")),
            },
        },
    }
