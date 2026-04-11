from __future__ import annotations

from ui.app import create_app_state
from ui.config import UIConfig
from ui.core_integration import build_system_state
from ui.layouts.operations_view import build_operations_view
from ui.reasoning.context_builder import build_reasoning_context
from ui.reasoning.reasoner import generate_reasoned_response

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
    assert view["zones"]["gate"]["layout"] == "full_width_authority_banner"
    assert view["zones"]["gate"]["visual_emphasis"] == "maximum"
    assert view["zones"]["system_state"]["brightness"] == "55%"
    assert view["zones"]["gate"]["content"]["authority_level"] == "SUPPRESSED"
    assert view["zones"]["gate"]["content"]["reality_status"] == "No confirmed change"
    assert view["zones"]["system_state"]["title"] == "System Context"
    assert view["zones"]["system_state"]["content"]["gate_coupling"]["decision"] == "SUPPRESS"


def test_operations_view_preserves_existing_reasoning_gate_context_when_gate_omitted() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.45,
            "relational_stability_score": 0.65,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    existing_gate = {
        "decision": "ADMIT",
        "reason": "Existing gate context from caller.",
        "doctrine_version": "doctrine.v2",
        "confidence_label": "high",
    }

    view = build_operations_view(
        system_state,
        reasoning_context={"gate_decision": existing_gate, "recent_admitted_events": []},
    )

    gate_reference = view["zones"]["reasoning"]["content"]["gate_reference"]
    assert gate_reference["decision"] == "ADMIT"
    assert gate_reference["reason"] == "Existing gate context from caller."
    assert gate_reference["doctrine_version"] == "doctrine.v2"
    assert gate_reference["confidence"] == "high"


def test_gate_banner_surfaces_reality_status_takeaway_intensity_and_timestamp() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.45,
            "relational_stability_score": 0.65,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "ADMIT",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
        "confidence_label": "high",
        "persistence_minutes": 75,
        "transition": {
            "delta_drift": 0.31,
            "delta_stability": -0.28,
            "type": "INSTABILITY",
        },
    }

    view = build_operations_view(system_state, gate_decision=gate_decision)
    gate_content = view["zones"]["gate"]["content"]

    assert gate_content["reality_status"] == "Change is real"
    assert gate_content["transition_type"] == "INSTABILITY"
    assert gate_content["transition_intensity"] == "HIGH"
    assert gate_content["operator_takeaway_label"] == "Operator Takeaway"
    assert gate_content["operator_takeaway"] == "System has entered a high-intensity instability transition."
    assert gate_content["risk_direction"] == "DEGRADING"
    assert gate_content["trajectory_statement"] == "System is progressing away from stable operating conditions."
    assert (
        gate_content["if_sustained_statement"]
        == "If sustained, this condition indicates: Potential transition into a new operating regime."
    )
    assert gate_content["timestamp_display"] == "Change evaluated at: 2026-04-10T00:00:00+00:00"


def test_gate_banner_impact_layer_for_suppressed_signal_is_stable() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.2,
            "relational_stability_score": 0.8,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "SUPPRESS",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
        "transition": {
            "delta_drift": 0.02,
            "delta_stability": 0.04,
            "type": "STABLE",
        },
    }

    view = build_operations_view(system_state, gate_decision=gate_decision)
    gate_content = view["zones"]["gate"]["content"]

    assert gate_content["risk_direction"] == "STABLE"
    assert gate_content["trajectory_statement"] == "No sustained directional change detected."
    assert (
        gate_content["if_sustained_statement"]
        == "If sustained, this condition indicates: No expected change in system behavior."
    )


def test_gate_banner_impact_layer_for_void_signal_is_uncertain() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.2,
            "relational_stability_score": 0.8,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "VOID",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
        "transition": {
            "delta_drift": 0.15,
            "delta_stability": -0.2,
            "type": "INSTABILITY",
        },
    }

    view = build_operations_view(system_state, gate_decision=gate_decision)
    gate_content = view["zones"]["gate"]["content"]

    assert gate_content["risk_direction"] == "UNCERTAIN"
    assert gate_content["trajectory_statement"] == "Signal coherence insufficient to determine system direction."
    assert (
        gate_content["if_sustained_statement"]
        == "If sustained, this condition indicates: Insufficient evidence to project system evolution."
    )




def test_gate_banner_treats_admitted_no_history_transition_as_uncertain() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.8,
            "relational_stability_score": 0.2,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "ADMIT",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
        "transition": {
            "delta_drift": 0.0,
            "delta_stability": 0.0,
            "delta_coherence": 0.0,
            "type": "STABLE",
        },
    }

    view = build_operations_view(system_state, gate_decision=gate_decision)
    gate_content = view["zones"]["gate"]["content"]

    assert gate_content["risk_direction"] == "UNCERTAIN"
    assert gate_content["operator_takeaway"] == "Admitted change detected, but directional trend remains uncertain."
    assert gate_content["trajectory_statement"] == "Directional trajectory remains uncertain under current evidence."
    assert (
        gate_content["if_sustained_statement"]
        == "If sustained, this condition indicates: Insufficient evidence to project system evolution."
    )

def test_gate_banner_normalizes_non_string_transition_type() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.45,
            "relational_stability_score": 0.65,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "ADMIT",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
        "transition": {
            "delta_drift": 0.02,
            "delta_stability": 0.03,
            "type": 7,
        },
    }

    view = build_operations_view(system_state, gate_decision=gate_decision)
    gate_content = view["zones"]["gate"]["content"]

    assert gate_content["transition_type"] == "STABLE"
    assert gate_content["operator_takeaway"] == "System has entered a low-intensity stable transition."


def test_suppress_gate_blocks_current_admit_record_and_event_point() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.4,
            "relational_stability_score": 0.6,
            "event_admitted": True,
        },
        {
            "timestamp": "2026-04-10T00:05:00+00:00",
            "structural_drift_score": 0.82,
            "relational_stability_score": 0.21,
            "event_admitted": True,
        },
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "SUPPRESS",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
        "transition": {"type": "INSTABILITY", "delta_drift": 0.33, "delta_stability": -0.3},
    }
    reasoning_context = build_reasoning_context(system_state, rows, gate_decision=gate_decision)

    view = build_operations_view(
        system_state,
        records=rows,
        reasoning_context=reasoning_context,
        gate_decision=gate_decision,
    )

    record_entries = view["zones"]["record"]["content"]["entries"]
    assert all(not (entry["timestamp"] == rows[-1]["timestamp"] and entry["gate_decision"] == "ADMIT") for entry in record_entries)

    admitted_points = view["zones"]["system_state"]["content"]["phase_layers"]["admitted_events"]
    assert all(point["t"] != rows[-1]["timestamp"] for point in admitted_points)


def test_suppress_gate_avoids_admitted_structural_change_wording() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.72,
            "relational_stability_score": 0.28,
        }
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {"decision": "SUPPRESS", "doctrine_version": "doctrine.v1"}
    reasoning_context = build_reasoning_context(system_state, rows, gate_decision=gate_decision)

    response = generate_reasoned_response("What is happening right now?", reasoning_context)

    operational = response["sections"]["Operational Implication"]
    assert all("admitted structural change" not in line.lower() for line in operational)
    assert any("directional evidence" in line.lower() for line in operational)


def test_admit_gate_aligns_record_event_and_reasoning() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.35,
            "relational_stability_score": 0.66,
            "event_admitted": False,
        },
        {
            "timestamp": "2026-04-10T00:05:00+00:00",
            "structural_drift_score": 0.84,
            "relational_stability_score": 0.2,
            "event_admitted": True,
        },
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {
        "decision": "ADMIT",
        "doctrine_version": "doctrine.v1",
        "timestamp": rows[-1]["timestamp"],
    }
    reasoning_context = build_reasoning_context(system_state, rows, gate_decision=gate_decision)
    view = build_operations_view(
        system_state,
        records=rows,
        reasoning_context=reasoning_context,
        gate_decision=gate_decision,
    )

    record_entries = view["zones"]["record"]["content"]["entries"]
    assert any(entry["timestamp"] == rows[-1]["timestamp"] and entry["gate_decision"] == "ADMIT" for entry in record_entries)

    admitted_points = view["zones"]["system_state"]["content"]["phase_layers"]["admitted_events"]
    assert any(point["t"] == rows[-1]["timestamp"] for point in admitted_points)

    response = generate_reasoned_response("What is happening right now?", reasoning_context)
    operational = response["sections"]["Operational Implication"]
    assert any("admitted structural change" in line.lower() for line in operational)


def test_system_view_encodes_admitted_and_suppressed_event_visual_styles() -> None:
    rows = [
        {
            "timestamp": "2026-04-10T00:00:00+00:00",
            "structural_drift_score": 0.35,
            "relational_stability_score": 0.66,
            "event_admitted": False,
        },
        {
            "timestamp": "2026-04-10T00:05:00+00:00",
            "structural_drift_score": 0.84,
            "relational_stability_score": 0.2,
            "event_admitted": True,
        },
    ]
    system_state = build_system_state(rows, config=UIConfig())
    gate_decision = {"decision": "ADMIT", "doctrine_version": "doctrine.v1", "timestamp": rows[-1]["timestamp"]}
    reasoning_context = build_reasoning_context(system_state, rows, gate_decision=gate_decision)
    view = build_operations_view(
        system_state,
        records=rows,
        reasoning_context=reasoning_context,
        gate_decision=gate_decision,
    )

    phase_layers = view["zones"]["system_state"]["content"]["phase_layers"]
    admitted_points = phase_layers["admitted_events"]
    suppressed_points = phase_layers["suppressed_events"]
    encoding = phase_layers["event_visual_encoding"]

    assert encoding["continuity_anchor"] == "trajectory.path"
    assert admitted_points and suppressed_points
    assert admitted_points[0]["event_type"] == "admitted"
    assert suppressed_points[0]["event_type"] == "suppressed"
    assert admitted_points[0]["glow"] > suppressed_points[0]["glow"]
    assert admitted_points[0]["opacity"] > suppressed_points[0]["opacity"]
