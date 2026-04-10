from __future__ import annotations

from typing import Any

from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate
from ui.reasoning import build_reasoning_context


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

    gate_decision = evaluate_gate(latest)

    reasoning_context: dict[str, Any]
    if rows:
        system_state = build_system_state(rows, config=UIConfig())
        reasoning_context = build_reasoning_context(system_state, rows)
    else:
        reasoning_context = {
            "current_state": {
                "timestamp": None,
                "regime": None,
                "system_health": None,
                "confidence": None,
                "drift": None,
                "stability": None,
            },
            "recent_admitted_events": [],
            "transition_point": None,
            "drift_summary": "No admitted drift evidence is available.",
            "stability_summary": "No admitted stability evidence is available.",
            "top_contributing_signals": None,
            "chart_replay_summary": None,
        }

    reasoning_context["gate_decision"] = {
        "decision": gate_decision.get("decision"),
        "reason": gate_decision.get("refusal_reason") or gate_decision.get("explanation"),
        "doctrine_version": gate_decision.get("doctrine_version"),
    }

    return {
        "summary": {
            "timestamp": latest.get("timestamp"),
            "system_health": latest.get("system_health"),
            "confidence": latest.get("confidence_score"),
            "drift": latest.get("structural_drift_score"),
            "stability": latest.get("relational_stability_score"),
            "regime": latest.get("regime_name"),
        },
        "gate_decision": gate_decision,
        "reasoning_context": reasoning_context,
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

    def load():
        data = create_app_state([])
        return (
            data["gate_decision"],
            data["summary"],
            data["reasoning_context"],
        )

    with gr.Blocks() as app:
        gr.Markdown("# Neraium — Operations Surface")

        gate = gr.JSON(label="Gate Decision")
        system = gr.JSON(label="System State")
        reasoning = gr.JSON(label="Evidence-Bound Reasoning")

        btn = gr.Button("Load System")
        btn.click(fn=load, outputs=[gate, system, reasoning])

    return app
