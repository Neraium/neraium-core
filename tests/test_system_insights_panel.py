from __future__ import annotations

from ui.components.gate_decision_card import render_gate_decision_card


def test_system_insights_uses_clear_terminology_and_state_mapping() -> None:
    card = render_gate_decision_card(
        {
            "decision": "ADMIT",
            "timestamp": "2026-07-24T07:12:15+00:00",
            "persistence_minutes": 42,
            "transition": {
                "type": "TRANSITION",
                "delta_drift": 0.18,
                "delta_stability": -0.12,
                "delta_coherence": -0.03,
            },
        }
    )

    insights = card["system_insights"]
    assert insights["decision"] == "Decision: Accepted"
    assert insights["state_transition"] == "State Transition: Confirmed"
    assert insights["system_state"] == "Drifting"
    assert insights["timestamp_display"] == "Jul 24, 07:12:15 AM"
    assert insights["insight_text"] == "Drift increasing but below alert threshold"


def test_system_insights_expands_ramp_up_phase_context() -> None:
    card = render_gate_decision_card(
        {
            "decision": "SUPPRESS",
            "phase_label": "Ramp Up",
            "transition": {"type": "STABLE", "delta_drift": 0.0, "delta_stability": 0.0},
        }
    )

    assert card["system_insights"]["phase_context"] == "Ramp Up (System stabilizing)"
