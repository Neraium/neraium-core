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
    operator_question: str = "What is happening right now?",
    gate_decision: dict[str, Any] | None = None,
) -> dict[str, object]:
    context = dict(reasoning_context or {})
    gate = gate_decision or {}

    context["gate_decision"] = {
        "decision": gate.get("decision"),
        "reason": gate.get("refusal_reason") or gate.get("explanation") or gate.get("reason"),
        "doctrine_version": gate.get("doctrine_version"),
        "confidence_label": gate.get("confidence_label"),
    }

    return {
        "mode": "operations",
        "viewport": "mobile" if width_px < 760 else "desktop",
        "hierarchy": "Observe → Evaluate → Admit / Suppress / Admissibility Void → Explain",
        "zones": {
            "gate": render_gate_decision_card(gate),
            "system_state": render_structural_flow_viz(state, gate_decision=gate),
            "reasoning": render_operational_reasoning_panel(operator_question, context),
            "record": render_event_ledger(context.get("recent_admitted_events")),
        },
    }
