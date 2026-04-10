from __future__ import annotations

from typing import Any

from ui.components import render_gate_decision_card, render_operational_reasoning_panel, render_structural_flow_viz
from ui.core_integration import SystemState


def build_operations_view(
    state: SystemState,
    *,
    width_px: int = 1440,
    reasoning_context: dict[str, Any] | None = None,
    operator_question: str = "What is happening right now?",
    gate_decision: dict[str, Any] | None = None,
) -> dict[str, object]:
    context = reasoning_context or {}
    return {
        "mode": "operations",
        "viewport": "mobile" if width_px < 760 else "desktop",
        "zones": {
            "gate": render_gate_decision_card(gate_decision),
            "system_state": render_structural_flow_viz(state, gate_decision=gate_decision),
            "reasoning": render_operational_reasoning_panel(operator_question, context),
            "record": {
                "component": "event_ledger",
                "events": context.get("recent_admitted_events", []),
            },
        },
    }
