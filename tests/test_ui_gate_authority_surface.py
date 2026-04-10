from __future__ import annotations

from ui.app import create_app_state
from ui.config import UIConfig
from ui.core_integration import build_system_state
from ui.layouts.operations_view import build_operations_view


def test_create_app_state_includes_gate_decision_and_reasoning_gate_context() -> None:
    row = {
        "timestamp": "2026-04-10T00:00:00+00:00",
        "asset_id": "unit-7",
        "structural_drift_score": 0.8,
        "relational_stability_score": 0.2,
        "coherence_score": 0.8,
        "snr_score": 2.0,
        "persistence_minutes": 30,
        "corroborating_signal_count": 3,
        "candidate_assertion": "Observed system degradation.",
    }

    state = create_app_state([row])

    assert state["gate_decision"]["decision"] == "ADMIT"
    assert state["reasoning_context"]["gate_decision"]["decision"] == "ADMIT"
    assert state["reasoning_context"]["gate_decision"]["doctrine_version"]


def test_operations_view_places_gate_zone_first() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.45,
            "relational_stability_score": 0.65,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {"decision": "SUPPRESS", "doctrine_version": "doctrine.v1", "timestamp": rows[-1]["timestamp"]}
    view = build_operations_view(
        system_state,
        reasoning_context={"recent_admitted_events": []},
        gate_decision=gate_decision,
    )

    assert list(view["zones"].keys()) == ["gate", "system_state", "reasoning", "record"]
    assert view["zones"]["gate"]["decision"] == "SUPPRESS"
    assert view["zones"]["system_state"]["gate_coupling"]["decision"] == "SUPPRESS"
