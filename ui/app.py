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
            "snr_score": 1.65,
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
            "snr_score": 1.62,
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
            "snr_score": 2.1,
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
            "bg": "#f5f6f7",
            "fg": "#1f2937",
            "border": "#9ca3af",
            "subtle": "#6b7280",
            "chip_bg": "#eef0f2",
        },
        "ADMITTED": {
            "bg": "#111827",
            "fg": "#f9fafb",
            "border": "#60a5fa",
            "subtle": "#cbd5e1",
            "chip_bg": "rgba(96,165,250,0.12)",
        },
        "VOID": {
            "bg": "#eef2ff",
            "fg": "#1e1b4b",
            "border": "#a5b4fc",
            "subtle": "#4f46e5",
            "chip_bg": "#e3e8ff",
        },
    }.get(
        authority_level,
        {
            "bg": "#f8fafc",
            "fg": "#0f172a",
            "border": "#cbd5e1",
            "subtle": "#334155",
            "chip_bg": "#f1f5f9",
        },
    )

    label = escape(str(gate_card.get("label") or "UNSPECIFIED GATE DECISION"))
    authority_badge = escape(str(gate_card.get("authority_badge") or authority_level))
    authority_statement = escape(str(gate_card.get("authority_statement") or "No authority statement available."))
    risk_direction = escape(str(gate_card.get("risk_direction") or "UNCERTAIN"))
    transition_type = escape(str(gate_card.get("transition_type") or "STABLE"))
    confidence = escape(str(gate_card.get("confidence") or "LOW"))
    operator_takeaway = escape(str(gate_card.get("operator_takeaway") or "No operator takeaway available."))
    if_sustained = escape(
        str(
            gate_card.get("if_sustained_statement")
            or "If sustained, this condition indicates: Insufficient evidence to project system evolution."
        )
    )
    timestamp_display = escape(str(gate_card.get("timestamp_display") or "Change evaluated at: unknown"))
    doctrine_version = escape(str(gate_card.get("doctrine_version") or "unknown"))
    reason = gate_card.get("reason") or gate_card.get("refusal_reason")
    reason_html = ""
    if reason:
        reason_html = (
            f"<div style=\"margin-top:8px;font-size:11px;line-height:1.4;color:{style['subtle']};\">"
            f"Context: {escape(str(reason))}</div>"
        )

    chip_style = (
        "font-size:10px;font-weight:780;letter-spacing:0.08em;text-transform:uppercase;"
        "padding:4px 8px;border-radius:999px;border:1px solid {border};"
        "background:{chip_bg};color:{subtle};"
    ).format(border=style["border"], chip_bg=style["chip_bg"], subtle=style["subtle"])

    return f"""
<div style="border:1px solid {style["border"]};border-radius:14px;padding:18px 18px 14px;background:{style["bg"]};color:{style["fg"]};box-shadow:0 1px 2px rgba(15,23,42,0.08);">
  <div style="font-size:32px;font-weight:900;line-height:1.05;letter-spacing:0.01em;text-transform:uppercase;">{label}</div>

  <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
    <span style="{chip_style}">{authority_badge}</span>
    <span style="{chip_style}">Risk {risk_direction}</span>
    <span style="{chip_style}">Transition {transition_type}</span>
    <span style="{chip_style}">Confidence {confidence}</span>
  </div>

  <div style="margin-top:14px;font-size:16px;font-weight:650;line-height:1.4;">{authority_statement}</div>
  <div style="margin-top:10px;font-size:15px;font-weight:800;line-height:1.35;">{operator_takeaway}</div>

  <div style="margin-top:12px;padding-top:10px;border-top:1px solid {style["border"]};font-size:12px;line-height:1.45;color:{style["fg"]};">
    {if_sustained}
  </div>

  <div style="margin-top:10px;display:flex;justify-content:space-between;gap:10px;font-size:11px;color:{style["subtle"]};">
    <span>{timestamp_display}</span>
    <span>Doctrine {doctrine_version}</span>
  </div>
  {reason_html}
</div>
""".strip()


def _render_system_context_html(system_zone: dict[str, Any]) -> str:
    content = system_zone.get("content") if isinstance(system_zone, dict) else {}
    timeline = system_zone.get("timeline_strip") if isinstance(system_zone, dict) else {}
    if not isinstance(content, dict):
        content = {}
    if not isinstance(timeline, dict):
        timeline = {}

    current = content.get("current_position") if isinstance(content.get("current_position"), dict) else {}
    trajectory = content.get("trajectory") if isinstance(content.get("trajectory"), dict) else {}
    phase_layers = content.get("phase_layers") if isinstance(content.get("phase_layers"), dict) else {}
    gate_coupling = content.get("gate_coupling") if isinstance(content.get("gate_coupling"), dict) else {}
    sequence = timeline.get("sequence") if isinstance(timeline.get("sequence"), list) else []

    def _count_points(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    def _to_points(value: Any) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        if isinstance(value, list):
            for raw in value:
                if not isinstance(raw, dict):
                    continue
                x = raw.get("x")
                y = raw.get("y")
                if isinstance(x, (float, int)) and isinstance(y, (float, int)):
                    points.append({"x": float(x), "y": float(y)})
        return points

    def _fmt(value: Any) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return "n/a"

    stable_count = _count_points(phase_layers.get("stable_baseline"))
    transition_count = _count_points(phase_layers.get("transition"))
    reorg_count = _count_points(phase_layers.get("reorganization"))
    admitted_markers = _count_points(phase_layers.get("admitted_events"))
    suppressed_markers = _count_points(phase_layers.get("suppressed_events"))
    trajectory_points_list = _to_points(trajectory.get("path"))
    trajectory_points = len(trajectory_points_list)

    def _to_svg_xy(point: dict[str, float], width: float = 880, height: float = 320, pad: float = 22.0) -> tuple[float, float]:
        x = pad + point["x"] * (width - pad * 2)
        y = pad + (1 - point["y"]) * (height - pad * 2)
        return x, y

    path_cmd = ""
    if trajectory_points_list:
        path_parts = []
        for idx, point in enumerate(trajectory_points_list):
            px, py = _to_svg_xy(point)
            prefix = "M" if idx == 0 else "L"
            path_parts.append(f"{prefix}{px:.2f},{py:.2f}")
        path_cmd = " ".join(path_parts)

    def _points_markup(points: list[dict[str, float]], color: str, radius: float, opacity: float) -> str:
        circles = []
        for point in points:
            px, py = _to_svg_xy(point)
            circles.append(
                f"<circle cx='{px:.2f}' cy='{py:.2f}' r='{radius:.2f}' fill='{color}' fill-opacity='{opacity:.2f}' />"
            )
        return "".join(circles)

    stable_points = _to_points(phase_layers.get("stable_baseline"))
    transition_points = _to_points(phase_layers.get("transition"))
    reorg_points = _to_points(phase_layers.get("reorganization"))
    admitted_points = _to_points(phase_layers.get("admitted_events"))
    suppressed_points = _to_points(phase_layers.get("suppressed_events"))
    current_xy = (
        _to_svg_xy({"x": float(current["x"]), "y": float(current["y"])})
        if isinstance(current.get("x"), (float, int)) and isinstance(current.get("y"), (float, int))
        else None
    )
    timeline_markup = "".join(
        (
            "<li style='margin:2px 0;'>"
            f"<strong>{escape(str(step.get('label') or step.get('stage') or 'UNKNOWN'))}</strong>"
            f" · t={escape(str(step.get('t') or 'n/a'))}"
            "</li>"
        )
        for step in sequence
        if isinstance(step, dict)
    )

    primary_surface = (
        f"""
<div style="margin-top:12px;border:1px solid #253048;border-radius:12px;background:radial-gradient(120% 100% at 20% 0%, #172447 0%, #080d1a 55%, #05070f 100%);padding:12px;">
  <svg viewBox="0 0 880 320" width="100%" height="260" role="img" aria-label="Structural trajectory context">
    <defs>
      <linearGradient id="traj" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#64D2FF" stop-opacity="0.78" />
        <stop offset="100%" stop-color="#B374FF" stop-opacity="0.96" />
      </linearGradient>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="c"/><feMerge><feMergeNode in="c"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <rect x="0" y="0" width="880" height="320" fill="none" stroke="#28344f" stroke-opacity="0.7" rx="10" />
    <path d="{path_cmd}" fill="none" stroke="url(#traj)" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" filter="url(#glow)" />
    {_points_markup(stable_points, "#7A8AA8", 2.2, 0.46)}
    {_points_markup(transition_points, "#9088FF", 2.8, 0.66)}
    {_points_markup(reorg_points, "#FFAA80", 3.2, 0.76)}
    {_points_markup(suppressed_points, "#9F8D86", 2.6, 0.34)}
    {_points_markup(admitted_points, "#73FFE1", 4.1, 0.9)}
    {f"<line x1='{current_xy[0]:.2f}' y1='{current_xy[1]:.2f}' x2='{current_xy[0] + 36:.2f}' y2='{current_xy[1] - 24:.2f}' stroke='#BFD1FF' stroke-opacity='0.72' stroke-width='1.6'/>" if current_xy else ""}
    {f"<circle cx='{current_xy[0]:.2f}' cy='{current_xy[1]:.2f}' r='11' fill='#89A8FF' fill-opacity='0.15'/>" if current_xy else ""}
    {f"<circle cx='{current_xy[0]:.2f}' cy='{current_xy[1]:.2f}' r='5.8' fill='#DDE7FF' fill-opacity='0.95'/>" if current_xy else ""}
  </svg>
</div>
"""
        if path_cmd
        else """
<div style="margin-top:12px;border:1px dashed #334155;border-radius:12px;padding:14px;background:#090f1d;color:#dbeafe;font-size:13px;">
  Structural renderer unavailable; fallback summary is shown for this frame.
</div>
"""
    )

    timeline_pills = "".join(
        (
            "<span style='display:inline-block;padding:4px 8px;border-radius:999px;border:1px solid #334155;"
            "background:#111827;color:#e5e7eb;font-size:11px;font-weight:700;margin:3px 4px 0 0;'>"
            f"{escape(str(step.get('label') or step.get('stage') or 'UNKNOWN'))} · t={escape(str(step.get('t') or 'n/a'))}"
            "</span>"
        )
        for step in sequence
        if isinstance(step, dict)
    )

    return f"""
<div style="border:1px solid #27354d;border-radius:14px;padding:16px;background:#060b16;color:#f8fafc;box-shadow:0 8px 30px rgba(15,23,42,0.35);">
  <div style="font-size:13px;font-weight:900;letter-spacing:0.06em;text-transform:uppercase;color:#dbeafe;">System Context</div>
  <div style="margin-top:6px;font-size:12px;color:#cbd5e1;">Structural/3D navigation field with trajectory continuity, phase layering, and gate coupling.</div>
  {primary_surface}
  <div style="margin-top:12px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;font-size:12px;line-height:1.45;">
    <div style="background:#0b1220;border:1px solid #233149;border-radius:10px;padding:10px;"><strong style="color:#f8fafc;">Current position</strong><br>x={_fmt(current.get("x"))} · y={_fmt(current.get("y"))} · t={escape(str(current.get("t", "n/a")))} · phase={escape(str(current.get("phase") or "unknown"))}</div>
    <div style="background:#0b1220;border:1px solid #233149;border-radius:10px;padding:10px;"><strong style="color:#f8fafc;">Trajectory</strong><br>{trajectory_points} path points with continuity dominance around current position.</div>
    <div style="background:#0b1220;border:1px solid #233149;border-radius:10px;padding:10px;"><strong style="color:#f8fafc;">Phase layers</strong><br>stable={stable_count} · transition={transition_count} · reorganization={reorg_count}</div>
    <div style="background:#0b1220;border:1px solid #233149;border-radius:10px;padding:10px;"><strong style="color:#f8fafc;">Gate coupling</strong><br>admitted={admitted_markers} · suppressed={suppressed_markers} · decision={escape(str(gate_coupling.get("decision") or "unknown"))}</div>
  </div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid #334155;">
    <div style="font-size:12px;font-weight:800;color:#e2e8f0;">Timeline strip</div>
    <div style="margin-top:4px;">{timeline_pills or "<span style='font-size:12px;color:#cbd5e1;'>n/a</span>"}</div>
    <ul style="margin:8px 0 0 16px;padding:0;color:#cbd5e1;font-size:12px;">{timeline_markup or "<li>n/a</li>"}</ul>
  </div>
</div>
""".strip()


def _render_reasoning_html(reasoning_panel: dict[str, Any]) -> str:
    panel = reasoning_panel if isinstance(reasoning_panel, dict) else {}
    facts = panel.get("observed_facts")
    if not isinstance(facts, list):
        facts = []
    inference = panel.get("inference") if isinstance(panel.get("inference"), dict) else {}
    sections = inference.get("sections") if isinstance(inference.get("sections"), dict) else {}
    grounded = (
        sections.get("Inference / Grounded Answer")
        or sections.get("Inference")
        or inference.get("final_response")
        or inference.get("answer")
    )
    if isinstance(grounded, list):
        grounded = " ".join(str(item) for item in grounded if item is not None).strip()
    gate_ref = panel.get("gate_reference") if isinstance(panel.get("gate_reference"), dict) else {}

    facts_markup = "".join(f"<li>{escape(str(fact))}</li>" for fact in facts)
    return f"""
<div style="border:1px solid #2b3a52;border-radius:14px;padding:16px;background:#0a1120;color:#f8fafc;box-shadow:0 8px 26px rgba(2,6,23,0.45);">
  <div style="font-size:13px;font-weight:900;letter-spacing:0.06em;text-transform:uppercase;color:#dbeafe;">Evidence-Bound Reasoning</div>
  <div style="margin-top:12px;display:grid;gap:10px;">
    <section style="border:1px solid #334155;border-radius:10px;padding:10px;background:#0d1629;">
      <div style="font-size:12px;font-weight:800;color:#f1f5f9;">Observed Facts</div>
      <ul style="margin:6px 0 0 18px;padding:0;color:#e2e8f0;font-size:13px;line-height:1.5;">{facts_markup or "<li>No observed facts available.</li>"}</ul>
    </section>
    <section style="border:1px solid #334155;border-radius:10px;padding:10px;background:#0d1629;">
      <div style="font-size:12px;font-weight:800;color:#f1f5f9;">Grounded Answer</div>
      <div style="margin-top:6px;font-size:13px;line-height:1.55;color:#e2e8f0;">{escape(str(grounded or "No grounded answer available."))}</div>
    </section>
    <section style="border:1px solid #334155;border-radius:10px;padding:10px;background:#0d1629;">
      <div style="font-size:12px;font-weight:800;color:#f1f5f9;">Operational Implication</div>
      <div style="margin-top:6px;font-size:13px;line-height:1.5;color:#e2e8f0;">{escape(str(panel.get("operational_implication") or "No implication available."))}</div>
    </section>
    <section style="border:1px solid #334155;border-radius:10px;padding:10px;background:#0d1629;">
      <div style="font-size:12px;font-weight:800;color:#f1f5f9;">Gate Outcome / Gate Reference</div>
      <div style="margin-top:6px;font-size:13px;color:#e2e8f0;">{escape(str(panel.get("gate_outcome") or gate_ref.get("decision") or "unknown"))} · Doctrine {escape(str(gate_ref.get("doctrine_version") or "unknown"))}</div>
    </section>
  </div>
</div>
""".strip()


def _render_record_html(record_panel: dict[str, Any]) -> str:
    panel = record_panel if isinstance(record_panel, dict) else {}
    entries = panel.get("entries")
    if not isinstance(entries, list):
        entries = []

    entry_markup = "".join(
        (
            "<article style='border:1px solid #334155;border-radius:10px;padding:12px;margin-top:10px;background:#0c1628;'>"
            "<div style='display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;'>"
            f"<div style='font-size:12px;color:#dbeafe;'><strong style='color:#f8fafc;'>{escape(str(entry.get('timestamp') or 'n/a'))}</strong></div>"
            f"<div style='font-size:11px;font-weight:800;border:1px solid #3b82f6;color:#dbeafe;border-radius:999px;padding:2px 8px;background:rgba(59,130,246,0.14);'>{escape(str(entry.get('gate_decision') or 'unknown'))}</div>"
            "</div>"
            f"<div style='font-size:13px;margin-top:8px;color:#e2e8f0;line-height:1.45;'><strong style='color:#f8fafc;'>Summary:</strong> {escape(str(entry.get('summary') or 'n/a'))}</div>"
            "<div style='margin-top:8px;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;'>"
            f"<div style='font-size:12px;color:#cbd5e1;'><strong style='color:#f1f5f9;'>Previous state</strong><br>{escape(str(entry.get('previous_state_summary') or 'n/a'))}</div>"
            f"<div style='font-size:12px;color:#cbd5e1;'><strong style='color:#f1f5f9;'>Current state</strong><br>{escape(str(entry.get('current_state_summary') or 'n/a'))}</div>"
            f"<div style='font-size:12px;color:#cbd5e1;'><strong style='color:#f1f5f9;'>Transition type</strong><br>{escape(str(entry.get('transition_type') or 'n/a'))}</div>"
            f"<div style='font-size:12px;color:#cbd5e1;'><strong style='color:#f1f5f9;'>Doctrine</strong><br>{escape(str(entry.get('doctrine_version') or 'unknown'))}</div>"
            "</div>"
            f"<div style='margin-top:8px;font-size:11px;color:#94a3b8;'>Record hash: {escape(str(entry.get('hash') or 'hash:pending'))}</div>"
            "</article>"
        )
        for entry in entries
        if isinstance(entry, dict)
    )
    return f"""
<div style="border:1px solid #2b3a52;border-radius:14px;padding:16px;background:#0a1120;color:#f8fafc;box-shadow:0 8px 26px rgba(2,6,23,0.45);">
  <div style="font-size:13px;font-weight:900;letter-spacing:0.06em;text-transform:uppercase;color:#dbeafe;">Evidence Record</div>
  <div style="margin-top:6px;font-size:12px;color:#cbd5e1;">Operational ledger of admitted/suppressed evidence transitions.</div>
  {entry_markup or "<div style='margin-top:10px;font-size:12px;color:#e2e8f0;'>No evidence entries available.</div>"}
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
        system_html = _render_system_context_html(surface["zones"]["system_state"])
        reasoning_html = _render_reasoning_html(surface["zones"]["reasoning"]["content"])
        record_html = _render_record_html(surface["zones"]["record"]["content"])
        return (
            gate_html,
            system_html,
            reasoning_html,
            record_html,
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
        system = gr.HTML(label="System Context", value=initial_system)
        reasoning = gr.HTML(label="Evidence-Bound Reasoning", value=initial_reasoning)
        record = gr.HTML(label="Evidence Record", value=initial_record)

        demo_step.change(fn=load_operations_surface, inputs=[demo_step], outputs=[gate, system, reasoning, record])

    return app
