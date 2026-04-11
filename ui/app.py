from __future__ import annotations

from html import escape
from typing import Any

from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate
from ui.layouts.operations_view import build_operations_view
from ui.reasoning import build_reasoning_context


def load_builtin_demo_rows() -> list[dict[str, Any]]:
    """Return a compact operations demo sequence for local UI startup."""
    return [
        {
            "timestamp": "2026-04-10T00:00:00Z",
            "regime_name": "baseline",
            "system_health": "nominal",
            "confidence_score": 0.71,
            "structural_drift_score": 0.22,
            "relational_stability_score": 0.84,
            "coherence_score": 0.87,
            "persistence_minutes": 0,
            "corroborating_signal_count": 0,
            "event_admitted": False,
            "transition_type": "STABLE",
            "evidence_summary": "Stable baseline: drift low, stability and coherence high.",
        },
        {
            "timestamp": "2026-04-10T00:05:00Z",
            "regime_name": "transition_watch",
            "system_health": "watch",
            "confidence_score": 0.68,
            "structural_drift_score": 0.58,
            "relational_stability_score": 0.44,
            "coherence_score": 0.56,
            "persistence_minutes": 12,
            "corroborating_signal_count": 1,
            "event_admitted": False,
            "transition_type": "TRANSITION",
            "evidence_summary": "Rising drift with weak corroboration; transition signal is currently suppressed.",
        },
        {
            "timestamp": "2026-04-10T00:12:00Z",
            "regime_name": "reorganization_candidate",
            "system_health": "degraded",
            "confidence_score": 0.79,
            "structural_drift_score": 0.77,
            "relational_stability_score": 0.27,
            "coherence_score": 0.74,
            "persistence_minutes": 44,
            "corroborating_signal_count": 3,
            "event_admitted": True,
            "transition_type": "REORGANIZATION",
            "evidence_summary": "Persistence and corroboration now qualify a coherent reorganization; transition admitted.",
        },
    ]


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
        "transition": {
            "type": "STABLE",
            "delta_drift": 0.0,
            "delta_stability": 0.0,
            "delta_coherence": 0.0,
        },
    }


def create_app_state(records=None):
    """
    Minimal integration-safe builder used by INTEGRATION_GUIDE.py.
    Accepts either:
    - a list of records
    - a single record dict
    - empty input (loads built-in demo rows)
    - unknown invalid input (falls back to suppressed gate decision)
    """
    if isinstance(records, list) and len(records) > 0:
        latest = records[-1]
        previous = records[-2] if len(records) > 1 else None
        rows = records
    elif isinstance(records, dict):
        latest = records
        previous = None
        rows = [records]
    elif records is None or records == []:
        rows = load_builtin_demo_rows()
        latest = rows[-1]
        previous = rows[-2] if len(rows) > 1 else None
    else:
        rows = []
        latest = {}
        previous = None

    replay_story = {
        "state_transitions": [str(row.get("regime_name") or row.get("state") or "unknown") for row in rows],
        "drift_trend": [
            float(row.get("structural_drift_score"))
            for row in rows
            if isinstance(row.get("structural_drift_score"), (int, float))
        ],
    }

    summary = {
        "timestamp": latest.get("timestamp"),
        "site_id": latest.get("site_id"),
        "asset_id": latest.get("asset_id"),
        "system_health": latest.get("system_health"),
        "confidence": latest.get("confidence_score"),
        "drift": latest.get("structural_drift_score"),
        "stability": latest.get("relational_stability_score"),
        "regime": latest.get("regime_name"),
        "replay_story": replay_story,
    }

    if rows:
        system_state = build_system_state(rows, config=UIConfig())
        gate_decision = evaluate_gate(latest, previous, system_state)
        if not isinstance(gate_decision, dict) or not gate_decision:
            gate_decision = _fallback_gate_decision()
        reasoning_context: dict[str, Any] = build_reasoning_context(system_state, rows, gate_decision=gate_decision)
        if not isinstance(reasoning_context, dict) or not reasoning_context:
            reasoning_context = {
                "current_state": summary,
                "gate_decision": {
                    "decision": gate_decision.get("decision") or "SUPPRESS",
                    "reason": gate_decision.get("reason") or gate_decision.get("refusal_reason"),
                },
                "recent_admitted_events": [],
                "operational_implication": "Demo fallback reasoning context.",
            }
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


def _render_gate_decision_html(gate_card: dict[str, Any]) -> str:
    authority_level = str(gate_card.get("authority_level") or "VOID").upper()
    style = {
        "SUPPRESSED": {
            "bg": "#f3f4f6",
            "fg": "#374151",
            "border": "#9ca3af",
            "subtle": "#6b7280",
        },
        "ADMITTED": {
            "bg": "#111827",
            "fg": "#f9fafb",
            "border": "#60a5fa",
            "subtle": "#d1d5db",
        },
        "VOID": {
            "bg": "#eef2ff",
            "fg": "#312e81",
            "border": "#a5b4fc",
            "subtle": "#4f46e5",
        },
    }.get(
        authority_level,
        {"bg": "#f8fafc", "fg": "#0f172a", "border": "#cbd5e1", "subtle": "#334155"},
    )

    label = escape(str(gate_card.get("label") or "UNSPECIFIED GATE DECISION"))
    subheader = escape(authority_level)
    trajectory = escape(str(gate_card.get("trajectory_statement") or "No trajectory statement available."))
    risk_direction = escape(str(gate_card.get("risk_direction") or "UNCERTAIN"))
    if_sustained = escape(
        str(
            gate_card.get("if_sustained_statement")
            or "If sustained, this condition indicates: Insufficient evidence to project system evolution."
        )
    )
    confidence = escape(str(gate_card.get("confidence") or "LOW"))

    return f"""
<div style="border:1px solid {style["border"]};border-left:8px solid {style["border"]};border-radius:12px;padding:16px;background:{style["bg"]};color:{style["fg"]};">
  <div style="font-size:24px;font-weight:800;letter-spacing:0.02em;">{label}</div>
  <div style="margin-top:4px;font-size:12px;font-weight:700;letter-spacing:0.08em;color:{style["subtle"]};">{subheader}</div>
  <div style="margin-top:14px;display:grid;gap:10px;">
    <div><div style="font-size:11px;text-transform:uppercase;opacity:0.8;">Trajectory</div><div style="font-size:15px;font-weight:600;">{trajectory}</div></div>
    <div><div style="font-size:11px;text-transform:uppercase;opacity:0.8;">Risk Direction</div><div style="font-size:15px;font-weight:600;">{risk_direction}</div></div>
    <div><div style="font-size:11px;text-transform:uppercase;opacity:0.8;">If Sustained</div><div style="font-size:15px;font-weight:600;">{if_sustained}</div></div>
    <div><div style="font-size:11px;text-transform:uppercase;opacity:0.8;">Confidence</div><div style="font-size:15px;font-weight:600;">{confidence}</div></div>
  </div>
</div>
""".strip()


def create_gradio_app():
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError("Gradio is not installed")

    demo_rows = load_builtin_demo_rows()

    demo_steps = {
        "Step 1: Baseline": demo_rows[:1],
        "Step 2: Transition / Suppress": demo_rows[:2],
        "Step 3: Reorganization / Admit": demo_rows[:3],
    }

    def load_operations_surface(step_label: str):
        active_rows = demo_steps.get(step_label) or demo_steps["Step 3: Reorganization / Admit"]
        app_state = create_app_state(active_rows)
        system_state = build_system_state(active_rows, config=UIConfig())
        latest = active_rows[-1] if active_rows else {}
        previous = active_rows[-2] if len(active_rows) > 1 else None

        gate_decision = app_state.get("gate_decision") if isinstance(app_state.get("gate_decision"), dict) else {}
        if not gate_decision:
            gate_decision = evaluate_gate(latest, previous, system_state)

        reasoning_context = app_state.get("reasoning_context") if isinstance(app_state.get("reasoning_context"), dict) else {}
        if not reasoning_context:
            reasoning_context = build_reasoning_context(system_state, active_rows, gate_decision=gate_decision)

        surface = build_operations_view(
            system_state,
            records=active_rows,
            reasoning_context=reasoning_context,
            gate_decision=gate_decision,
        )
        gate_content = surface["zones"]["gate"]["content"]
        gate_html = _render_gate_decision_html(gate_content if isinstance(gate_content, dict) else {})
        return (
            gate_html,
            surface["zones"]["system_state"],
            surface["zones"]["reasoning"]["content"],
            surface["zones"]["record"]["content"],
        )

    initial_gate, initial_system, initial_reasoning, initial_record = load_operations_surface(
        "Step 3: Reorganization / Admit"
    )

    with gr.Blocks() as app:
        gr.Markdown("# Neraium — Gate-Centered Operations Surface")

        demo_step = gr.Radio(
            choices=list(demo_steps.keys()),
            value="Step 3: Reorganization / Admit",
            label="Demo Progression",
        )

        gate = gr.HTML(label="Gate Decision", value=initial_gate)
        system = gr.JSON(label="System Context", value=initial_system)
        reasoning = gr.JSON(label="Evidence-Bound Reasoning", value=initial_reasoning)
        record = gr.JSON(label="Evidence Record", value=initial_record)

        demo_step.change(fn=load_operations_surface, inputs=[demo_step], outputs=[gate, system, reasoning, record])

    return app
