from __future__ import annotations

from typing import Any

from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate
from ui.demo_data import load_greenhouse_demo_records
from ui.reasoning import build_reasoning_context


def _trend_label(delta: float, *, positive: str, negative: str, neutral: str = "stable") -> str:
    if delta > 0.02:
        return positive
    if delta < -0.02:
        return negative
    return neutral


def _summarize_replay_story(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "state_transitions": 0,
            "drift_trend": "stable",
            "stability_trend": "stable",
            "confidence_trend": "stable",
            "health_trajectory": "flat",
            "story_headline": "No greenhouse replay frames loaded.",
        }

    first = rows[0]
    last = rows[-1]
    drift_delta = float(last.get("structural_drift_score", 0.0)) - float(first.get("structural_drift_score", 0.0))
    stability_delta = float(last.get("relational_stability_score", 1.0)) - float(first.get("relational_stability_score", 1.0))
    confidence_delta = float(last.get("confidence_score", 0.0)) - float(first.get("confidence_score", 0.0))

    states = [str(row.get("state") or "unknown") for row in rows]
    transitions = sum(1 for idx in range(1, len(states)) if states[idx] != states[idx - 1])

    health_map = {"nominal": 0, "watch": 1, "degraded": 2, "critical": 3}
    first_health = health_map.get(str(first.get("system_health") or "nominal"), 0)
    last_health = health_map.get(str(last.get("system_health") or "nominal"), 0)
    health_delta = last_health - first_health

    drift_trend = _trend_label(drift_delta, positive="rising", negative="easing")
    stability_trend = _trend_label(stability_delta, positive="recovering", negative="weakening")
    confidence_trend = _trend_label(confidence_delta, positive="increasing", negative="decreasing")
    health_trajectory = _trend_label(health_delta, positive="worsening", negative="improving", neutral="flat")

    headline = (
        f"{transitions} state transitions observed; drift {drift_trend}, "
        f"stability {stability_trend}, confidence {confidence_trend}."
    )
    return {
        "state_transitions": transitions,
        "drift_trend": drift_trend,
        "stability_trend": stability_trend,
        "confidence_trend": confidence_trend,
        "health_trajectory": health_trajectory,
        "story_headline": headline,
    }


def create_app_state(records=None):
    """
    Minimal integration-safe builder used by INTEGRATION_GUIDE.py.
    Accepts either:
    - a list of records
    - a single record dict
    - empty / unknown input (loads greenhouse demo defaults)
    """
    if isinstance(records, list) and len(records) > 0:
        latest = records[-1]
        rows = records
    elif isinstance(records, dict):
        latest = records
        rows = [records]
    else:
        rows = load_greenhouse_demo_records(limit=96)
        latest = rows[-1] if rows else {}

    gate_decision = evaluate_gate(latest)
    replay_story = _summarize_replay_story(rows)

    reasoning_context: dict[str, Any]
    if rows:
        system_state = build_system_state(rows, config=UIConfig())
        reasoning_context = build_reasoning_context(system_state, rows)
        if isinstance(reasoning_context.get("chart_replay_summary"), dict):
            reasoning_context["chart_replay_summary"].update(replay_story)
            reasoning_context["chart_replay_summary"]["regime_span"] = {
                "start": str(rows[0].get("regime_name") or rows[0].get("state") or "unknown"),
                "end": str(rows[-1].get("regime_name") or rows[-1].get("state") or "unknown"),
            }
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
            "chart_replay_summary": replay_story,
        }

    reasoning_context["gate_decision"] = {
        "decision": gate_decision.get("decision"),
        "reason": gate_decision.get("refusal_reason") or gate_decision.get("explanation"),
        "doctrine_version": gate_decision.get("doctrine_version"),
    }

    return {
        "summary": {
            "timestamp": latest.get("timestamp"),
            "site_id": latest.get("site_id"),
            "asset_id": latest.get("asset_id"),
            "state": latest.get("state"),
            "system_health": latest.get("system_health"),
            "confidence": latest.get("confidence_score"),
            "drift": latest.get("structural_drift_score"),
            "stability": latest.get("relational_stability_score"),
            "regime": latest.get("regime_name"),
            "replay_story": replay_story,
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
        gr.Markdown("# Neraium — Greenhouse Structural Replay")
        gr.Markdown("High-signal demo view: state transitions, drift/stability movement, health/confidence trends.")

        gate = gr.JSON(label="Gate Decision")
        system = gr.JSON(label="System State + Replay Story")
        reasoning = gr.JSON(label="Evidence-Bound Reasoning")

        btn = gr.Button("Load Greenhouse Demo")
        btn.click(fn=load, outputs=[gate, system, reasoning])

    return app
