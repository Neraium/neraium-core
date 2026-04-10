from __future__ import annotations

from typing import Any

from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate
from ui.layouts.operations_view import build_operations_view
from ui.reasoning import build_reasoning_context


def _fallback_gate_decision() -> dict[str, Any]:
    return {
        "decision": "SUPPRESS",
        "doctrine_version": "unknown",
        "criteria_results": {},
        "refusal_reason": "No admitted telemetry row is available for evaluation.",
        "explanation": "Suppressed: no current admitted telemetry is available.",
        "observed_facts": ["No current record was provided to the UI integration layer."],
        "uncertainty_notes": ["Gate evaluation used conservative fallback behavior."],
        "candidate_assertion_allowed": False,
        "confidence_label": "low",
        "timestamp": None,
    }


def create_app_state(records):
    """
    Minimal integration-safe builder used by INTEGRATION_GUIDE.py.
    Accepts either:
    - a list of records
    - a single record dict
    - empty / unknown input
    """
    if isinstance(records, list) and len(records) > 0:
        latest = records[-1]
        rows = records
    elif isinstance(records, dict):
        latest = records
        rows = [records]
    else:
        latest = {}
        rows = []

    summary = {
        "timestamp": latest.get("timestamp"),
        "system_health": latest.get("system_health"),
        "confidence": latest.get("confidence_score"),
        "drift": latest.get("structural_drift_score"),
        "stability": latest.get("relational_stability_score"),
        "regime": latest.get("regime_name"),
    }

    if rows:
        system_state = build_system_state(rows, config=UIConfig())
        gate_decision = evaluate_gate(latest, system_state)
        reasoning_context: dict[str, Any] = build_reasoning_context(system_state, rows, gate_decision=gate_decision)
    else:
        gate_decision = _fallback_gate_decision()
        reasoning_context = {
            "current_state": {
                "timestamp": None,
                "regime": None,
                "system_health": None,
                "confidence": None,
                "drift": None,
                "stability": None,
            },
            "gate_decision": {
                "decision": gate_decision.get("decision"),
                "reason": gate_decision.get("refusal_reason"),
                "doctrine_version": gate_decision.get("doctrine_version"),
                "confidence_label": gate_decision.get("confidence_label"),
            },
            "recent_admitted_events": [],
            "transition_point": None,
            "drift_summary": "No admitted drift evidence is available.",
            "stability_summary": "No admitted stability evidence is available.",
            "top_contributing_signals": None,
            "chart_replay_summary": None,
        }

    return {
        "summary": summary,
        "reasoning_context": reasoning_context,
        "gate_decision": gate_decision,
        "realtime": {
            "enabled": False,
        },
    }


def create_ui_model(data):
    return {
        "summary": data[-1] if isinstance(data, list) and data else (data if isinstance(data, dict) else {}),
        "realtime": {"enabled": False},
    }


def create_gradio_app():
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError("Gradio is not installed")

    def load_operations_surface():
        sample_rows = [
            {
                "timestamp": "2026-04-10T00:00:00Z",
                "regime_name": "baseline",
                "system_health": "nominal",
                "confidence_score": 0.72,
                "structural_drift_score": 0.34,
                "relational_stability_score": 0.71,
            },
            {
                "timestamp": "2026-04-10T00:05:00Z",
                "regime_name": "transition",
                "system_health": "watch",
                "confidence_score": 0.68,
                "structural_drift_score": 0.58,
                "relational_stability_score": 0.43,
                "coherence_score": 0.62,
                "snr_score": 1.4,
                "persistence_minutes": 10,
                "corroborating_signal_count": 1,
                "event_admitted": False,
                "evidence_summary": "Drift elevated with limited corroboration in current window.",
            },
        ]

        app_state = create_app_state(sample_rows)
        system_state = build_system_state(sample_rows, config=UIConfig())
        surface = build_operations_view(
            system_state,
            reasoning_context=app_state["reasoning_context"],
            gate_decision=app_state["gate_decision"],
        )
        return (
            surface["zones"]["gate"],
            app_state["summary"],
            surface["zones"]["reasoning"],
            surface["zones"]["record"],
        )

    with gr.Blocks() as app:
        gr.Markdown("# Neraium — Gate-Centered Operations Surface")

        gate = gr.JSON(label="Gate Decision")
        system = gr.JSON(label="System State")
        reasoning = gr.JSON(label="Evidence-Bound Reasoning")
        record = gr.JSON(label="Recent Record")

        btn = gr.Button("Load Operations Surface")
        btn.click(fn=load_operations_surface, outputs=[gate, system, reasoning, record])

    return app
