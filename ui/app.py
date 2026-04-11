from __future__ import annotations

from html import escape
from pathlib import Path
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
        "SUPPRESSED": {"accent": "#ffb66e", "badge": "#ff9a3d"},
        "ADMITTED": {"accent": "#62eaff", "badge": "#2ecbff"},
        "VOID": {"accent": "#8ea4ff", "badge": "#8ea4ff"},
    }.get(authority_level, {"accent": "#8ea4ff", "badge": "#8ea4ff"})

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
            "<div class=\"ner-context-line\">"
            f"Context: {escape(str(reason))}</div>"
        )

    def _chip(label_text: str, value: str) -> str:
        return (
            f'<span class="ner-chip">'
            f'<span class="ner-chip-label">{escape(label_text)}</span>'
            f'<span class="ner-chip-value">{escape(value)}</span>'
            f"</span>"
        )

    return f"""
<div class="ner-panel ner-verdict-card" style="--verdict-accent:{style["accent"]};--verdict-badge:{style["badge"]};">
  <div class="ner-eyebrow">Gate Verdict</div>
  <div class="ner-verdict-main">{label}</div>
  <div class="ner-verdict-subtitle">{authority_statement}</div>
  <div class="ner-verdict-takeaway">{operator_takeaway}</div>
  <div class="ner-chip-row">
    {_chip("Authority", authority_badge)}
    {_chip("Risk", risk_direction)}
    {_chip("Transition", transition_type)}
    {_chip("Confidence", confidence)}
    {_chip("Doctrine", doctrine_version)}
  </div>
  <div class="ner-verdict-projection">
    {if_sustained}
  </div>
  <div class="ner-meta-row">
    <span>{timestamp_display}</span>
    <span class="ner-badge">{authority_badge}</span>
  </div>
  {reason_html}
</div>
""".strip()


def _render_system_context_html(system_zone: dict[str, Any]) -> str:
    content = system_zone.get("content") if isinstance(system_zone, dict) else {}
    timeline_data = system_zone.get("timeline_strip") if isinstance(system_zone, dict) else {}
    if not isinstance(content, dict):
        content = {}
    if not isinstance(timeline_data, dict):
        timeline_data = {}

    def _f(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _cl(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    # Canvas layout
    W, H = 900, 290
    PX, PY = 52, 38
    IW = W - 2 * PX
    IH = H - 2 * PY

    def sx(x: Any) -> float:
        return round(PX + _cl(_f(x), 0.0, 1.0) * IW, 2)

    def sy(y: Any) -> float:
        return round(PY + _cl(_f(y), 0.0, 1.0) * IH, 2)

    trajectory = content.get("trajectory") if isinstance(content.get("trajectory"), dict) else {}
    current = content.get("current_position") if isinstance(content.get("current_position"), dict) else {}
    vel = content.get("velocity_vector") if isinstance(content.get("velocity_vector"), dict) else {}
    phase_layers = content.get("phase_layers") if isinstance(content.get("phase_layers"), dict) else {}
    gate_coupling = content.get("gate_coupling") if isinstance(content.get("gate_coupling"), dict) else {}
    projected = content.get("projected_forward_region") if isinstance(content.get("projected_forward_region"), dict) else {}

    path = trajectory.get("path") if isinstance(trajectory.get("path"), list) else []
    fading_tail = trajectory.get("fading_tail") if isinstance(trajectory.get("fading_tail"), list) else []
    decision = str(gate_coupling.get("decision") or "SUPPRESS").upper()
    phase = str(current.get("phase") or "coherence")

    parts: list[str] = []

    # SVG defs: filters, gradients, arrowhead
    parts.append(
        f'<defs>'
        f'<linearGradient id="tg" x1="{PX}" y1="0" x2="{W - PX}" y2="0" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0%" stop-color="#5CCAFF" stop-opacity="0.18"/>'
        f'<stop offset="50%" stop-color="#7F87FF" stop-opacity="0.52"/>'
        f'<stop offset="100%" stop-color="#B35EFF" stop-opacity="0.82"/>'
        f'</linearGradient>'
        f'<radialGradient id="ph" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="#00EEFF" stop-opacity="0.42"/>'
        f'<stop offset="55%" stop-color="#5CCAFF" stop-opacity="0.10"/>'
        f'<stop offset="100%" stop-color="#5CCAFF" stop-opacity="0.0"/>'
        f'</radialGradient>'
        f'<filter id="fxl" x="-100%" y="-100%" width="300%" height="300%">'
        f'<feGaussianBlur in="SourceGraphic" stdDeviation="7" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        f'<filter id="flg" x="-80%" y="-80%" width="260%" height="260%">'
        f'<feGaussianBlur in="SourceGraphic" stdDeviation="4" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        f'<filter id="fmd" x="-60%" y="-60%" width="220%" height="220%">'
        f'<feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        f'<filter id="fsm" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feGaussianBlur in="SourceGraphic" stdDeviation="1.5" result="b"/>'
        f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
        f'<marker id="vh" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="6" markerHeight="5" orient="auto">'
        f'<path d="M0 0 L10 4 L0 8 Z" fill="#93C5FD" opacity="0.82"/>'
        f'</marker>'
        f'</defs>'
    )

    # Stability region background bands (vertical strips by drift/x)
    band_defs = [
        (0.0, 0.35, "rgba(96,211,211,0.055)", "rgba(96,211,211,0.32)", "STABLE"),
        (0.35, 0.65, "rgba(130,120,255,0.065)", "rgba(130,120,255,0.32)", "TRANSITION"),
        (0.65, 1.0, "rgba(255,140,100,0.075)", "rgba(255,140,100,0.32)", "DIVERGENCE"),
    ]
    for x0, x1, fill, label_color, label in band_defs:
        bx = sx(x0)
        bw = round(IW * (x1 - x0), 2)
        mid_x = sx((x0 + x1) / 2.0)
        parts.append(f'<rect x="{bx}" y="{PY}" width="{bw}" height="{IH}" fill="{fill}"/>')
        parts.append(
            f'<text x="{mid_x}" y="{PY + 15}" text-anchor="middle" '
            f'font-size="9" font-weight="700" fill="{label_color}" letter-spacing="0.09em">{label}</text>'
        )

    # Projected forward cone (faint forward uncertainty region)
    cone_samples = projected.get("samples") if isinstance(projected.get("samples"), list) else []
    cone_op = _cl(_f(projected.get("opacity"), 0.18), 0.0, 0.5)
    for p in cone_samples:
        if isinstance(p, dict):
            try:
                parts.append(
                    f'<circle cx="{sx(p.get("x", 0.5))}" cy="{sy(p.get("y", 0.5))}" '
                    f'r="6" fill="rgba(147,197,253,0.13)" opacity="{cone_op:.2f}"/>'
                )
            except (TypeError, ValueError):
                pass

    # Fading trajectory dots
    for pt in fading_tail:
        if isinstance(pt, dict):
            try:
                r = _cl(_f(pt.get("radius"), 2.0), 1.0, 8.0)
                op = _cl(_f(pt.get("opacity"), 0.3), 0.05, 1.0)
                color = str(pt.get("color") or "rgba(115,143,255,0.35)")
                parts.append(
                    f'<circle cx="{sx(pt.get("x", 0.5))}" cy="{sy(pt.get("y", 0.5))}" '
                    f'r="{r:.1f}" fill="{color}" opacity="{op:.3f}"/>'
                )
            except (TypeError, ValueError):
                pass

    # Trajectory polyline with gradient stroke
    if len(path) >= 2:
        pts_str = " ".join(
            f"{sx(p.get('x', 0.5))},{sy(p.get('y', 0.5))}"
            for p in path
            if isinstance(p, dict)
        )
        parts.append(
            f'<polyline points="{pts_str}" fill="none" stroke="url(#tg)" '
            f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="0.72" filter="url(#fsm)"/>'
        )

    # Phase layer: stable baseline (calm blue-grey)
    for p in (phase_layers.get("stable_baseline") or []):
        if isinstance(p, dict):
            try:
                parts.append(
                    f'<circle cx="{sx(p.get("x", 0.5))}" cy="{sy(p.get("y", 0.5))}" '
                    f'r="3.2" fill="rgba(110,125,157,0.45)" opacity="0.65"/>'
                )
            except (TypeError, ValueError):
                pass

    # Phase layer: transition (mobilizing purple)
    for p in (phase_layers.get("transition") or []):
        if isinstance(p, dict):
            try:
                parts.append(
                    f'<circle cx="{sx(p.get("x", 0.5))}" cy="{sy(p.get("y", 0.5))}" '
                    f'r="4.8" fill="rgba(143,135,255,0.62)" filter="url(#fsm)"/>'
                )
            except (TypeError, ValueError):
                pass

    # Phase layer: reorganization (hot orange-red)
    for p in (phase_layers.get("reorganization") or []):
        if isinstance(p, dict):
            try:
                parts.append(
                    f'<circle cx="{sx(p.get("x", 0.5))}" cy="{sy(p.get("y", 0.5))}" '
                    f'r="6" fill="rgba(255,158,120,0.72)" filter="url(#fmd)"/>'
                )
            except (TypeError, ValueError):
                pass

    # Suppressed event markers (faded, denied-reality treatment)
    for p in (phase_layers.get("suppressed_events") or []):
        if isinstance(p, dict):
            try:
                r = _cl(_f(p.get("radius"), 2.5), 1.0, 5.0)
                op = _cl(_f(p.get("opacity"), 0.18), 0.05, 0.35)
                parts.append(
                    f'<circle cx="{sx(p.get("x", 0.5))}" cy="{sy(p.get("y", 0.5))}" '
                    f'r="{r:.1f}" fill="rgba(159,141,134,0.38)" opacity="{op:.2f}"/>'
                )
            except (TypeError, ValueError):
                pass

    # Admitted event markers (bright teal, glowing)
    for p in (phase_layers.get("admitted_events") or []):
        if isinstance(p, dict):
            try:
                cx_e = sx(p.get("x", 0.5))
                cy_e = sy(p.get("y", 0.5))
                r = _cl(_f(p.get("radius"), 6.0), 3.0, 12.0)
                parts.append(
                    f'<circle cx="{cx_e}" cy="{cy_e}" r="{r + 4:.1f}" '
                    f'fill="none" stroke="rgba(115,255,225,0.28)" stroke-width="1.5" filter="url(#fmd)"/>'
                )
                parts.append(
                    f'<circle cx="{cx_e}" cy="{cy_e}" r="{r:.1f}" '
                    f'fill="rgba(115,255,225,0.92)" filter="url(#flg)"/>'
                )
            except (TypeError, ValueError):
                pass

    # Velocity vector arrow
    vel_origin = vel.get("origin") if isinstance(vel.get("origin"), dict) else {}
    vel_tip_pt = vel.get("tip") if isinstance(vel.get("tip"), dict) else {}
    vel_mag = abs(_f(vel.get("dx"), 0.0)) + abs(_f(vel.get("dy"), 0.0))
    if vel_origin and vel_tip_pt and vel_mag > 0.001:
        try:
            ox, oy = sx(vel_origin.get("x", 0.5)), sy(vel_origin.get("y", 0.5))
            tx, ty = sx(vel_tip_pt.get("x", 0.5)), sy(vel_tip_pt.get("y", 0.5))
            if abs(ox - tx) + abs(oy - ty) > 4:
                parts.append(
                    f'<line x1="{ox}" y1="{oy}" x2="{tx}" y2="{ty}" '
                    f'stroke="#93C5FD" stroke-width="1.8" stroke-linecap="round" '
                    f'opacity="0.74" marker-end="url(#vh)" filter="url(#fsm)"/>'
                )
        except (TypeError, ValueError):
            pass

    # Current position — dominant focus node, rendered last (on top)
    if current:
        try:
            cx_p = sx(current.get("x", 0.5))
            cy_p = sy(current.get("y", 0.5))
            halo_r = _cl(_f(current.get("halo_radius"), 0.08) * IW * 0.55, 14.0, 58.0)
            parts.append(f'<circle cx="{cx_p}" cy="{cy_p}" r="{halo_r:.1f}" fill="url(#ph)"/>')
            parts.append(
                f'<circle cx="{cx_p}" cy="{cy_p}" r="11" '
                f'fill="none" stroke="rgba(0,238,255,0.38)" stroke-width="2" filter="url(#fmd)"/>'
            )
            parts.append(
                f'<circle cx="{cx_p}" cy="{cy_p}" r="5.5" '
                f'fill="#00EEFF" filter="url(#fxl)" opacity="0.96"/>'
            )
        except (TypeError, ValueError):
            pass

    svg_body = "\n".join(parts)
    svg_html = f'<svg class="ner-system-canvas" viewBox="0 0 {W} {H}" width="100%">\n{svg_body}\n</svg>'

    # Timeline strip
    sequence = timeline_data.get("sequence") if isinstance(timeline_data.get("sequence"), list) else []
    stage_palette = {
        "admitted": "#62FFB3",
        "suppressed": "#FB923C",
        "transition": "#818CF8",
        "baseline": "#4B5563",
        "void": "#A78BFA",
    }
    timeline_html = ""
    if sequence:
        items: list[str] = []
        for i, step in enumerate(sequence):
            if not isinstance(step, dict):
                continue
            label = str(step.get("label") or step.get("stage") or "?")
            stage = str(step.get("stage") or "baseline").lower()
            color = stage_palette.get(stage, "#4B5563")
            connector = (
                f'<div style="flex:1;height:1px;background:rgba(255,255,255,0.07);margin:0 3px;align-self:center;"></div>'
                if i < len(sequence) - 1
                else ""
            )
            items.append(
                f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;">'
                f'<div style="width:7px;height:7px;border-radius:50%;background:{color};box-shadow:0 0 6px {color}66;"></div>'
                f'<span style="font-size:8.5px;font-weight:700;letter-spacing:0.05em;color:{color};text-transform:uppercase;">'
                f'{escape(label)}</span></div>{connector}'
            )
        timeline_html = (
            f'<div class="ner-timeline-strip">'
            f'<span class="ner-timeline-label">Timeline</span>'
            f'{"".join(items)}</div>'
        )

    # Header bar
    badge_colors = {"ADMIT": "#62FFB3", "SUPPRESS": "#FB923C", "VOID": "#A78BFA"}
    badge_color = badge_colors.get(decision, "#9CA3AF")
    header_html = (
        f'<div class="ner-panel-head">'
        f'<span class="ner-eyebrow">System Context</span>'
        f'<div class="ner-panel-meta">'
        f'<span class="ner-chip-mini">Phase {escape(phase.upper())}</span>'
        f'<span class="ner-chip-mini" style="color:{badge_color};border-color:{badge_color}66;">{escape(decision)}</span>'
        f'</div></div>'
    )

    return (
        f'<div class="ner-panel ner-system-panel">'
        f'{header_html}{svg_html}{timeline_html}</div>'
    )


def _render_reasoning_html(reasoning_panel: dict[str, Any]) -> str:
    panel = reasoning_panel if isinstance(reasoning_panel, dict) else {}
    facts = panel.get("observed_facts")
    if not isinstance(facts, list):
        facts = []
    inference = panel.get("inference") if isinstance(panel.get("inference"), dict) else {}
    sections = inference.get("sections") if isinstance(inference.get("sections"), dict) else {}

    raw_grounded = sections.get("Inference") or inference.get("final_response") or inference.get("answer")
    if isinstance(raw_grounded, list):
        grounded_text = " ".join(str(i) for i in raw_grounded if i is not None).strip()
    else:
        grounded_text = str(raw_grounded or "").strip()

    insufficient = sections.get("Insufficient Evidence") or []
    if isinstance(insufficient, list) and insufficient:
        insufficient_text = " ".join(str(i) for i in insufficient if i is not None).strip()
    else:
        insufficient_text = ""

    gate_ref = panel.get("gate_reference") if isinstance(panel.get("gate_reference"), dict) else {}
    gate_decision_val = str(panel.get("gate_outcome") or gate_ref.get("decision") or "SUPPRESS").upper()
    doctrine_val = escape(str(gate_ref.get("doctrine_version") or "unknown"))
    confidence_val = escape(str(gate_ref.get("confidence") or "unknown"))
    op_impl = escape(str(panel.get("operational_implication") or "No implication available."))

    gate_badge_colors = {
        "ADMIT": ("rgba(98,255,179,0.12)", "#34D399", "#D1FAE5"),
        "SUPPRESS": ("rgba(251,146,60,0.12)", "#F97316", "#FED7AA"),
        "VOID": ("rgba(167,139,250,0.12)", "#A78BFA", "#EDE9FE"),
    }
    gb_bg, gb_border, gb_text = gate_badge_colors.get(gate_decision_val, ("rgba(156,163,175,0.1)", "#9CA3AF", "#E5E7EB"))

    def _section(title: str, body_html: str, accent: str = "#3B82F6") -> str:
        return (
            f'<div class="ner-reason-section">'
            f'<div class="ner-reason-title" style="color:{accent};">'
            f'{title}</div>'
            f'{body_html}'
            f'</div>'
        )

    # Observed Facts section
    if facts:
        facts_items = "".join(
            f'<div style="display:flex;gap:8px;margin-bottom:5px;">'
            f'<span style="color:#4B5563;margin-top:1px;flex-shrink:0;">·</span>'
            f'<span style="font-size:13px;line-height:1.55;color:#D1D5DB;">{escape(str(f))}</span>'
            f'</div>'
            for f in facts
        )
        facts_body = f'<div class="ner-facts-list">{facts_items}</div>'
    else:
        facts_body = '<div style="font-size:13px;color:#6B7280;font-style:italic;">No observed facts available.</div>'
    facts_section = _section("Observed Facts", facts_body, "#60A5FA")

    # Grounded Answer section
    if grounded_text:
        grounded_body = f'<div class="ner-reason-copy">{escape(grounded_text)}</div>'
    elif insufficient_text:
        grounded_body = (
            f'<div style="font-size:13px;line-height:1.55;color:#9CA3AF;font-style:italic;">'
            f'{escape(insufficient_text)}</div>'
        )
    else:
        grounded_body = '<div style="font-size:13px;color:#6B7280;font-style:italic;">No grounded answer available.</div>'
    grounded_section = _section("Grounded Answer", grounded_body, "#818CF8")

    # Operational Implication section
    impl_body = f'<div class="ner-reason-copy">{op_impl}</div>'
    impl_section = _section("Operational Implication", impl_body, "#34D399")

    # Gate Outcome section
    gate_body = (
        f'<div class="ner-reason-gate-row">'
        f'<span style="font-size:11px;font-weight:800;padding:4px 10px;border-radius:999px;'
        f'background:{gb_bg};color:{gb_text};border:1px solid {gb_border}44;letter-spacing:0.06em;'
        f'text-transform:uppercase;">{escape(gate_decision_val)}</span>'
        f'<span class="ner-subtle">Doctrine {doctrine_val}</span>'
        f'<span class="ner-subtle">Confidence: {confidence_val}</span>'
        f'</div>'
    )
    gate_section = _section("Gate Outcome", gate_body, "#FB923C")

    return (
        f'<div class="ner-panel">'
        f'<div class="ner-eyebrow">Reasoning Chain</div>'
        f'{facts_section}{grounded_section}{impl_section}{gate_section}'
        f'</div>'
    )


def _render_record_html(record_panel: dict[str, Any]) -> str:
    panel = record_panel if isinstance(record_panel, dict) else {}
    entries = panel.get("entries")
    if not isinstance(entries, list):
        entries = []

    decision_styles = {
        "ADMIT": {
            "bg": "rgba(52,211,153,0.08)",
            "border": "rgba(52,211,153,0.22)",
            "badge_bg": "rgba(52,211,153,0.14)",
            "badge_text": "#34D399",
            "badge_border": "rgba(52,211,153,0.38)",
            "ts_color": "#6EE7B7",
        },
        "SUPPRESS": {
            "bg": "rgba(251,146,60,0.06)",
            "border": "rgba(251,146,60,0.18)",
            "badge_bg": "rgba(251,146,60,0.12)",
            "badge_text": "#FB923C",
            "badge_border": "rgba(251,146,60,0.35)",
            "ts_color": "#FCA96A",
        },
        "VOID": {
            "bg": "rgba(167,139,250,0.06)",
            "border": "rgba(167,139,250,0.18)",
            "badge_bg": "rgba(167,139,250,0.12)",
            "badge_text": "#A78BFA",
            "badge_border": "rgba(167,139,250,0.35)",
            "ts_color": "#C4B5FD",
        },
    }
    default_style = {
        "bg": "rgba(75,85,99,0.06)",
        "border": "rgba(75,85,99,0.18)",
        "badge_bg": "rgba(75,85,99,0.12)",
        "badge_text": "#9CA3AF",
        "badge_border": "rgba(75,85,99,0.3)",
        "ts_color": "#9CA3AF",
    }

    transition_colors = {
        "REORGANIZATION": "#FB923C",
        "TRANSITION": "#818CF8",
        "STABLE": "#4B5563",
        "RECOVERY": "#34D399",
    }

    def _entry_card(entry: dict[str, Any]) -> str:
        raw_decision = str(entry.get("gate_decision") or "SUPPRESS").upper()
        st = decision_styles.get(raw_decision, default_style)
        ts = escape(str(entry.get("timestamp") or "n/a"))
        summary = escape(str(entry.get("summary") or "No evidence summary."))
        prev_state = escape(str(entry.get("previous_state_summary") or "n/a"))
        curr_state = escape(str(entry.get("current_state_summary") or "n/a"))
        raw_transition = str(entry.get("transition_type") or "STABLE").upper()
        transition_color = transition_colors.get(raw_transition, "#4B5563")

        return (
            f'<div class="ner-record-card" style="--card-border:{st["border"]};--card-bg:{st["bg"]};">'
            # Header row: timestamp + decision badge + transition type
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">'
            f'<span style="font-size:12px;font-weight:700;color:{st["ts_color"]};font-variant-numeric:tabular-nums;">'
            f'{ts}</span>'
            f'<span style="font-size:10px;font-weight:800;padding:2px 8px;border-radius:999px;'
            f'background:{st["badge_bg"]};color:{st["badge_text"]};border:1px solid {st["badge_border"]};'
            f'letter-spacing:0.07em;text-transform:uppercase;">{escape(raw_decision)}</span>'
            f'<span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px;'
            f'background:rgba(255,255,255,0.04);color:{transition_color};'
            f'border:1px solid {transition_color}33;letter-spacing:0.05em;text-transform:uppercase;">'
            f'{escape(raw_transition)}</span>'
            f'</div>'
            # Summary
            f'<div style="font-size:13px;line-height:1.55;color:#D1D5DB;margin-bottom:10px;">{summary}</div>'
            # State transition row
            f'<div class="ner-state-shift">'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:9px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
            f'color:#4B5563;margin-bottom:3px;">Previous</div>'
            f'<div style="color:#94A3B8;line-height:1.4;word-break:break-word;">{prev_state}</div>'
            f'</div>'
            f'<div style="color:#374151;font-size:14px;padding-top:14px;flex-shrink:0;">→</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div style="font-size:9px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
            f'color:#4B5563;margin-bottom:3px;">Current</div>'
            f'<div style="color:#CBD5E1;line-height:1.4;word-break:break-word;">{curr_state}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )

    if entries:
        cards_html = "".join(_entry_card(e) for e in entries if isinstance(e, dict))
    else:
        cards_html = (
            '<div style="margin-top:10px;font-size:13px;color:#4B5563;font-style:italic;">'
            'No evidence entries recorded.</div>'
        )

    return (
        f'<div class="ner-panel">'
        f'<div class="ner-eyebrow">Evidence Record</div>'
        f'{cards_html}'
        f'</div>'
    )



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

    def render_command_header(step_label: str) -> str:
        active_rows = demo_steps.get(step_label) or demo_steps["Step 3: Reorganization / Admit"]
        latest = active_rows[-1] if active_rows else {}
        confidence = f"{float(latest.get('confidence_score') or 0.0):.2f}"
        gate_state = "ADMIT" if latest.get("event_admitted") else "SUPPRESS"
        return f"""
            <div class="ner-command-header">
              <div class="ner-brand">
                <span class="ner-wordmark">NERAIUM</span>
                <span class="ner-env">GREENHOUSE / DEMO</span>
              </div>
              <div class="ner-header-metrics">
                <span>Doctrine v2026.04</span>
                <span>Confidence {escape(confidence)}</span>
                <span>Gate {escape(gate_state)}</span>
                <span>State LIVE DEMO</span>
              </div>
            </div>
            """

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
        header_html = render_command_header(step_label)
        return (
            header_html,
            gate_html,
            system_html,
            reasoning_html,
            record_html,
        )

    default_step = "Step 3: Reorganization / Admit"
    initial_header, initial_gate, initial_system, initial_reasoning, initial_record = load_operations_surface(default_step)

    css_path = Path(__file__).parent / "themes" / "neraium_dark.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    with gr.Blocks(css=css, theme=gr.themes.Base(), elem_classes=["ner-app"]) as app:
        header = gr.HTML(value=initial_header)

        with gr.Row(elem_classes=["ner-hero-row"]):
            with gr.Column(scale=8):
                gate = gr.HTML(value=initial_gate)
            with gr.Column(scale=4):
                demo_step = gr.Radio(
                    choices=list(demo_steps.keys()),
                    value=default_step,
                    label="Mission Flow",
                    elem_classes=["ner-flow-radio"],
                )

        system = gr.HTML(value=initial_system)
        reasoning = gr.HTML(value=initial_reasoning)
        record = gr.HTML(value=initial_record)

        demo_step.change(fn=load_operations_surface, inputs=[demo_step], outputs=[header, gate, system, reasoning, record])

    return app
