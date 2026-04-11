from __future__ import annotations

from html import escape
from pathlib import Path
import time
from typing import Any

from ui.config import UIConfig
from ui.core_integration import build_system_state, evaluate_gate
from ui.demo_data import load_greenhouse_demo_records
from ui.layouts.operations_view import build_operations_view
from ui.reasoning import build_reasoning_context
from ui.replay_timing import (
    ReplayPaceController,
    VerdictStabilizer,
    ReasoningStateTracker,
)


def load_builtin_demo_rows() -> list[dict[str, Any]]:
    """Return greenhouse replay sequence loaded from greenhouse_results_turbo.csv."""
    data_rows = load_greenhouse_demo_records(limit=180)
    if data_rows:
        return data_rows
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

    label = escape(str(gate_card.get("label") or authority_level))
    authority_statement = escape(str(gate_card.get("authority_statement") or ""))
    supporting_line = escape(str(gate_card.get("supporting_line") or ""))
    confidence = escape(str(gate_card.get("confidence") or "LOW"))
    transition_type = escape(str(gate_card.get("transition_type") or "STABLE"))
    risk_direction = escape(str(gate_card.get("risk_direction") or "UNCERTAIN"))
    ts_raw = str(gate_card.get("timestamp") or "")
    ts_display = escape(ts_raw[:19].replace("T", " ")) if ts_raw else "—"
    doctrine_version = escape(str(gate_card.get("doctrine_version") or "—"))

    def _chip(label_text: str, value: str) -> str:
        return (
            f'<span class="ner-chip">'
            f'<span class="ner-chip-label">{escape(label_text)}</span>'
            f'<span class="ner-chip-value">{escape(value)}</span>'
            f"</span>"
        )

    accent = style["accent"]
    badge = style["badge"]
    return f"""
<div class="ner-panel ner-verdict-card" style="--verdict-accent:{accent};--verdict-badge:{badge};">
  <div class="ner-verdict-main">{label}</div>
  <div class="ner-verdict-subtitle">{authority_statement}</div>
  <div class="ner-verdict-supporting">{supporting_line}</div>
  <div class="ner-chip-row">
    {_chip("Confidence", confidence)}
    {_chip("Phase", transition_type)}
    {_chip("Risk", risk_direction)}
  </div>
  <div class="ner-meta-row">
    <span style="font-size:11px;color:#4e6580;">{ts_display}</span>
    <span style="font-size:11px;color:#4e6580;">Engine {doctrine_version}</span>
  </div>
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
    W, H = 900, 300
    PX, PY = 56, 46
    IW = W - 2 * PX
    IH = H - 2 * PY

    def sx(x: Any) -> float:
        return round(PX + _cl(_f(x), 0.0, 1.0) * IW, 2)

    def sy(y: Any) -> float:
        return round(PY + _cl(_f(y), 0.0, 1.0) * IH, 2)

    trajectory = content.get("trajectory") if isinstance(content.get("trajectory"), dict) else {}
    gate_coupling = content.get("gate_coupling") if isinstance(content.get("gate_coupling"), dict) else {}
    replay_series = content.get("replay_series") if isinstance(content.get("replay_series"), list) else []
    decision = str(gate_coupling.get("decision") or "SUPPRESS").upper()
    path = trajectory.get("path") if isinstance(trajectory.get("path"), list) else []
    if replay_series:
        points = replay_series
    else:
        points = [{"index": idx, "signal": _f(p.get("x"), 0.0), "phase": "unknown"} for idx, p in enumerate(path) if isinstance(p, dict)]

    point_count = max(len(points), 1)
    chart_points: list[dict[str, Any]] = []
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            continue
        signal = _cl(_f(point.get("signal"), point.get("y")), 0.0, 1.0)
        chart_points.append(
            {
                "x": idx / max(point_count - 1, 1),
                "y": signal,
                "phase": str(point.get("phase") or "unknown").upper(),
                "timestamp": str(point.get("timestamp") or ""),
            }
        )
    current = chart_points[-1] if chart_points else {"x": 1.0, "y": 0.0, "phase": "UNKNOWN"}

    parts: list[str] = []
    parts.append(
        f'<defs>'
        f'<linearGradient id="tg" x1="{PX}" y1="{PY}" x2="{PX}" y2="{PY + IH}" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0%" stop-color="#38BDF8" stop-opacity="0.86"/>'
        f'<stop offset="100%" stop-color="#1D4ED8" stop-opacity="0.72"/>'
        f'</linearGradient>'
        f'</defs>'
    )

    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for y_tick in y_ticks:
        y_pos = sy(y_tick)
        parts.append(f'<line x1="{PX}" y1="{y_pos}" x2="{PX + IW}" y2="{y_pos}" stroke="rgba(148,163,184,0.16)" stroke-width="1"/>')
        parts.append(
            f'<text x="{PX - 10}" y="{y_pos + 4}" text-anchor="end" fill="rgba(203,213,225,0.88)" font-size="10">{y_tick:.2f}</text>'
        )

    phase_runs: list[tuple[int, int, str]] = []
    if chart_points:
        run_start = 0
        run_phase = chart_points[0]["phase"]
        for idx, point in enumerate(chart_points[1:], start=1):
            if point["phase"] != run_phase:
                phase_runs.append((run_start, idx - 1, run_phase))
                run_start = idx
                run_phase = point["phase"]
        phase_runs.append((run_start, len(chart_points) - 1, run_phase))
    phase_colors = {"STABLE": "rgba(30,64,175,0.14)", "TRANSITION": "rgba(91,33,182,0.12)", "REORGANIZATION": "rgba(194,65,12,0.12)"}
    for start_idx, end_idx, phase_name in phase_runs:
        x0 = sx(start_idx / max(point_count - 1, 1))
        x1 = sx(end_idx / max(point_count - 1, 1))
        if x1 > x0:
            parts.append(
                f'<rect x="{x0}" y="{PY}" width="{max(1.0, x1 - x0)}" height="{IH}" fill="{phase_colors.get(phase_name, "rgba(71,85,105,0.08)")}" />'
            )

    parts.append(f'<line x1="{PX}" y1="{PY + IH}" x2="{PX + IW}" y2="{PY + IH}" stroke="rgba(147,197,253,0.45)" stroke-width="1"/>')
    parts.append(f'<line x1="{PX}" y1="{PY}" x2="{PX}" y2="{PY + IH}" stroke="rgba(147,197,253,0.45)" stroke-width="1"/>')
    parts.append(f'<text x="{PX + IW - 4}" y="{PY + IH + 20}" text-anchor="end" fill="rgba(203,213,225,0.9)" font-size="11">Replay →</text>')
    parts.append(f'<text x="{PX - 8}" y="{PY - 8}" text-anchor="end" fill="rgba(203,213,225,0.9)" font-size="11">Signal</text>')

    if len(chart_points) >= 2:
        pts_str = " ".join(
            f"{sx(p.get('x', 0.5))},{sy(p.get('y', 0.5))}"
            for p in chart_points
            if isinstance(p, dict)
        )
        parts.append(
            f'<polyline points="{pts_str}" fill="none" stroke="url(#tg)" '
            f'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" '
            f'opacity="0.94"/>'
        )

    cx_p = sx(current.get("x", 1.0))
    cy_p = sy(current.get("y", 0.0))
    parts.append(f'<circle cx="{cx_p}" cy="{cy_p}" r="6.5" fill="#E2E8F0" stroke="#0EA5E9" stroke-width="2"/>')
    parts.append(f'<text x="{min(cx_p + 12, W - 12)}" y="{max(cy_p - 10, PY + 12)}" fill="#E2E8F0" font-size="11">Now</text>')

    svg_body = "\n".join(parts)
    svg_html = f'<svg class="ner-system-canvas" viewBox="0 0 {W} {H}" width="100%">\n{svg_body}\n</svg>'

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
            f'<span class="ner-timeline-label">Phase replay</span>'
            f'{"".join(items)}</div>'
        )

    header_html = (
        f'<div class="ner-panel-head">'
        f'<div style="display:flex;flex-direction:column;gap:2px;">'
        f'<span class="ner-eyebrow">Replay telemetry</span>'
        f'<span style="font-size:11px;color:#7c8ba8;">System path across {len(chart_points)} states · shows trajectory evidence</span>'
        f'</div>'
        f'</div>'
    )

    x = _f(current.get("x"), 0.0)
    y = _f(current.get("y"), 0.0)
    phase_label = escape(str(current.get("phase") or "UNKNOWN").upper())
    start_signal = _f(chart_points[0].get("y"), 0.0) if chart_points else 0.0
    context_row = (
        '<div class="ner-system-context-grid">'
        f'<div><span class="ner-context-label">Strength</span><span class="ner-context-value">{y:.3f}</span></div>'
        f'<div><span class="ner-context-label">Trajectory</span><span class="ner-context-value">{start_signal:.3f} → {y:.3f}</span></div>'
        f'<div><span class="ner-context-label">Regime</span><span class="ner-context-value">{phase_label}</span></div>'
        f'<div><span class="ner-context-label">Position</span><span class="ner-context-value">{int(round(x * max(point_count - 1, 0)))} / {max(point_count - 1, 0)}</span></div>'
        '</div>'
    )

    return (
        f'<div class="ner-panel ner-system-panel">'
        f'{header_html}{context_row}{svg_html}{timeline_html}</div>'
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

    op_impl = escape(str(panel.get("operational_implication") or "No implication available."))

    observed = escape(str(facts[0])) if facts else "No observed signal."
    assessment_source = grounded_text or insufficient_text or "No assessment available."
    implication_source = escape(str(panel.get("operational_implication") or "No implication available."))

    def _one_line(text: str) -> str:
        value = text.strip()
        if len(value) > 120:
            value = f"{value[:117].rstrip()}..."
        return value

    observed_line = _one_line(observed)
    assessment_line = _one_line(escape(assessment_source))
    implication_line = _one_line(implication_source)

    core_lines = (
        f'<div class="ner-core-lines">'
        f'<div class="ner-core-line"><span>Observed</span><strong>{observed_line}</strong></div>'
        f'<div class="ner-core-line"><span>Assessment</span><strong>{assessment_line}</strong></div>'
        f'<div class="ner-core-line"><span>Implication</span><strong>{implication_line}</strong></div>'
        f'</div>'
    )

    details_items = "".join(f"<li>{escape(str(f))}</li>" for f in facts[1:] if f)
    details_html = (
        f'<details class="ner-more-detail"><summary>More detail</summary>'
        f'<div class="ner-reason-copy">{escape(grounded_text or insufficient_text or "")}</div>'
        f'<div class="ner-reason-copy">{op_impl}</div>'
        f'<ul>{details_items}</ul>'
        f"</details>"
    )

    return (
        f'<div class="ner-panel">'
        f'<div class="ner-eyebrow">Reasoning</div>'
        f'{core_lines}{details_html}'
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
        decision_text = {"ADMIT": "CONFIRMED", "SUPPRESS": "OBSERVED", "VOID": "VOIDED"}.get(raw_decision, "OBSERVED")
        st = decision_styles.get(raw_decision, default_style)
        ts = escape(str(entry.get("timestamp") or "n/a"))
        summary = escape(str(entry.get("summary") or "No summary."))
        raw_transition = str(entry.get("transition_type") or "STABLE").upper()
        transition_color = transition_colors.get(raw_transition, "#4B5563")

        return (
            f'<div class="ner-record-card" style="--card-border:{st["border"]};--card-bg:{st["bg"]};">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
            f'<span style="font-size:11px;font-weight:700;color:{st["ts_color"]};font-variant-numeric:tabular-nums;">{ts}</span>'
            f'<span style="font-size:10px;font-weight:800;padding:2px 8px;border-radius:999px;'
            f'background:{st["badge_bg"]};color:{st["badge_text"]};border:1px solid {st["badge_border"]};'
            f'letter-spacing:0.07em;">{decision_text}</span>'
            f'<span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:999px;'
            f'background:rgba(255,255,255,0.04);color:{transition_color};'
            f'border:1px solid {transition_color}33;letter-spacing:0.05em;">{escape(raw_transition)}</span>'
            f'</div>'
            f'<div style="font-size:13px;line-height:1.45;color:#E5EDFF;margin-top:7px;">{summary}</div>'
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
        f'<div class="ner-eyebrow">Evidence</div>'
        f'{cards_html}'
        f'</div>'
    )



def _render_verdict_surface_html(gate_card: dict[str, Any], system_zone: dict[str, Any]) -> str:
    """Merge verdict card and trajectory into one visually connected surface."""
    gate_html = _render_gate_decision_html(gate_card)
    system_html = _render_system_context_html(system_zone)
    return f'<div class="ner-verdict-surface">{gate_html}{system_html}</div>'


def create_gradio_app():
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError("Gradio is not installed")

    demo_rows = load_builtin_demo_rows()
    total_steps = max(len(demo_rows), 1)
    playback_state = {"playing": False}

    # Replay stabilization for verdict and reasoning
    verdict_stabilizer = VerdictStabilizer(hysteresis_threshold=0.08)
    reasoning_tracker = ReasoningStateTracker(change_threshold=0.06)
    pace_controller = ReplayPaceController()

    def _rows_until(frame_index: int) -> list[dict[str, Any]]:
        idx = max(1, min(total_steps, int(frame_index)))
        return demo_rows[:idx]

    def render_command_header(frame_index: int) -> str:
        active_rows = _rows_until(frame_index)
        latest = active_rows[-1] if active_rows else {}
        confidence = f"{float(latest.get('confidence_score') or 0.0):.2f}"
        regime_raw = str(latest.get("system_phase") or latest.get("regime_name") or "unknown")
        regime_display = regime_raw.replace("_", " ").title()

        # Add phase progress indicator
        phase_label = "Baseline"
        transition_type = str(latest.get("transition_type", "STABLE")).upper()
        if transition_type in {"TRANSITION", "REORGANIZATION"}:
            phase_label = transition_type.title()

        return f"""
            <div class="ner-command-header">
              <div class="ner-brand">
                <span class="ner-wordmark">NERAIUM</span>
                <span class="ner-env">GREENHOUSE / DEMO</span>
              </div>
              <div class="ner-header-metrics">
                <span>Doctrine v2026.04</span>
                <span>Confidence {escape(confidence)}</span>
                <span>Phase {escape(phase_label)}</span>
                <span>Frame {int(frame_index)} / {int(total_steps)}</span>
              </div>
            </div>
            """

    def load_operations_surface(frame_index: int, apply_stability: bool = True):
        active_rows = _rows_until(frame_index)
        app_state = create_app_state(active_rows)
        system_state = build_system_state(active_rows, config=UIConfig())
        latest = active_rows[-1] if active_rows else {}
        previous = active_rows[-2] if len(active_rows) > 1 else None

        gate_decision = app_state.get("gate_decision") if isinstance(app_state.get("gate_decision"), dict) else {}
        if not gate_decision:
            gate_decision = evaluate_gate(latest, previous, system_state)

        # Apply verdict stability during replay to prevent flipping
        if apply_stability and isinstance(gate_decision, dict):
            drift_intensity = system_state.drift_intensity if system_state else 0.0
            gate_decision = verdict_stabilizer.apply_stability(
                gate_decision,
                signal_strength=drift_intensity,
            )

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
        gate_card = gate_content if isinstance(gate_content, dict) else {}
        verdict_html = _render_verdict_surface_html(gate_card, surface["zones"]["system_state"])
        reasoning_html = _render_reasoning_html(surface["zones"]["reasoning"]["content"])
        record_html = _render_record_html(surface["zones"]["record"]["content"])
        header_html = render_command_header(frame_index)
        return (
            header_html,
            verdict_html,
            reasoning_html,
            record_html,
        )

    def pause_playback() -> None:
        playback_state["playing"] = False

    def reset_playback() -> tuple[int, str, str, str, str]:
        pause_playback()
        # Reset stabilizers when resetting playback
        verdict_stabilizer.reset()
        reasoning_tracker.reset()
        header_html, verdict_html, reasoning_html, record_html = load_operations_surface(1, apply_stability=False)
        return 1, header_html, verdict_html, reasoning_html, record_html

    def autoplay(start_frame: int, speed_multiplier: float):
        playback_state["playing"] = True
        pace_controller.speed_multiplier = float(speed_multiplier or 1.0)
        frame = max(1, int(start_frame))

        while frame <= total_steps and playback_state["playing"]:
            # Calculate adaptive delay based on phase
            step_delay = pace_controller.get_step_delay(frame - 1, demo_rows)

            yield (frame, *load_operations_surface(frame, apply_stability=True))
            frame += 1

            # Sleep with calculated adaptive delay
            time.sleep(step_delay)

        playback_state["playing"] = False

    default_step = min(30, total_steps)
    initial_header, initial_verdict, initial_reasoning, initial_record = load_operations_surface(
        default_step, apply_stability=False
    )

    css_path = Path(__file__).parent / "themes" / "neraium_dark.css"
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    with gr.Blocks(css=css, theme=gr.themes.Base(), elem_classes=["ner-app"]) as app:
        playing_state = gr.State(False)
        header = gr.HTML(value=initial_header)
        verdict = gr.HTML(value=initial_verdict)

        with gr.Row(elem_classes=["ner-controls-row"]):
            frame_step = gr.Slider(minimum=1, maximum=total_steps, step=1, value=default_step, label="Frame", scale=4)
            speed = gr.Slider(minimum=0.1, maximum=1.5, step=0.1, value=0.6, label="Speed", scale=2)
            play_btn = gr.Button("Play", size="sm", scale=1)
            pause_btn = gr.Button("Pause", size="sm", scale=1)
            reset_btn = gr.Button("Reset", size="sm", scale=1)

        reasoning = gr.HTML(value=initial_reasoning)
        record = gr.HTML(value=initial_record)
        timer = gr.Timer(value=1.0, active=True)

        frame_step.change(fn=load_operations_surface, inputs=[frame_step], outputs=[header, verdict, reasoning, record])
        play_btn.click(fn=autoplay, inputs=[frame_step, speed], outputs=[frame_step, header, verdict, reasoning, record])
        pause_btn.click(fn=pause_playback)
        reset_btn.click(fn=reset_playback, outputs=[frame_step, header, verdict, reasoning, record])

    return app
