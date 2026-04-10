from __future__ import annotations

from typing import Any, Literal

from neraium_core.doctrine import (
    DEFAULT_DOCTRINE_V1,
    count_confirming_signals,
    evaluate_statement,
)

ASSISTANT_MODES = {
    "summary",
    "why_recommended",
    "what_changed",
    "pattern_similarity",
    "handoff",
}

REPORT_MODES = {
    "client_report",
    "technician_summary",
    "inspection_brief",
    "handoff_note",
}

AssistantMode = Literal["summary", "why_recommended", "what_changed", "pattern_similarity", "handoff"]
ReportMode = Literal["client_report", "technician_summary", "inspection_brief", "handoff_note"]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_assistant_context(
    *,
    current_state: dict[str, Any] | None,
    recent_history: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    state = _as_dict(current_state)
    history = [row for row in (recent_history or []) if isinstance(row, dict)]

    session = _as_dict(state.get("session"))
    risk = _as_dict(state.get("risk_assessment"))
    recommendation = _as_dict(state.get("operational_recommendation"))
    memory = _as_dict(state.get("memory_recall"))
    novelty = _as_dict(memory.get("novelty"))
    nearest = _as_dict(memory.get("nearest_match"))

    latest = history[0] if history else state
    previous = history[1] if len(history) > 1 else {}

    latest_risk = _as_dict(latest.get("risk_assessment"))
    previous_risk = _as_dict(previous.get("risk_assessment"))
    latest_events = _as_list(latest.get("events"))
    previous_events = set(_as_list(previous.get("events")))

    recent_timeline: list[dict[str, Any]] = []
    for row in history[:6]:
        row_risk = _as_dict(row.get("risk_assessment"))
        row_rec = _as_dict(row.get("operational_recommendation"))
        row_status = _as_dict(row_rec.get("status"))
        row_memory = _as_dict(row.get("memory_recall"))
        row_novelty = _as_dict(row_memory.get("novelty"))
        recent_timeline.append(
            {
                "cycle": row.get("cycle"),
                "timestamp": row.get("timestamp"),
                "risk_level": row_risk.get("risk_level"),
                "recommendation_available": bool(row_status.get("available")),
                "recommendation_confidence": row_rec.get("recommendation_confidence"),
                "events": _as_list(row.get("events")),
                "novelty": row_novelty.get("is_novel"),
            }
        )

    return {
        "current_state": {
            "timestamp": state.get("timestamp"),
            "cycle": state.get("cycle"),
            "run_id": state.get("run_id") or session.get("run_id"),
            "customer_id": state.get("customer_id"),
            "site_id": session.get("site_id"),
            "asset_id": session.get("asset_id"),
            "confidence": state.get("confidence"),
        },
        "risk": risk,
        "recommendation": recommendation,
        "explanation": state.get("explanation_text"),
        "events": _as_list(state.get("events")),
        "memory_recall": {
            "novelty": novelty,
            "nearest_match": nearest,
            "top_matches": _as_list(memory.get("top_matches")),
            "pattern_family": _as_dict(memory.get("pattern_family")),
        },
        "gate_decision": _as_dict(state.get("gate_decision")),
        "recent_changes": {
            "risk_level": {
                "current": latest_risk.get("risk_level"),
                "previous": previous_risk.get("risk_level"),
            },
            "new_events": [event for event in latest_events if event not in previous_events],
            "latest_timestamp": latest.get("timestamp"),
            "previous_timestamp": previous.get("timestamp"),
        },
        "recent_timeline": recent_timeline,
    }


def _grounded_text(prefix: str, value: Any, fallback: str) -> str:
    if value in (None, "", [], {}):
        return f"{prefix} {fallback}"
    return f"{prefix} {value}"


def _join_sections(title: str, sections: list[tuple[str, str]]) -> str:
    chunks = [title]
    for name, content in sections:
        chunks.append(f"\n{name}\n{content}")
    return "\n".join(chunks)


def _display_value(value: Any, *, fallback: str = "not available") -> str:
    if value in (None, "", [], {}):
        return fallback
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            if item in (None, "", [], {}):
                continue
            parts.append(f"{key}: {item}")
        return "; ".join(parts) if parts else fallback
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if item not in (None, "", [], {})]
        return "; ".join(cleaned) if cleaned else fallback
    return str(value)


def _format_list(items: Any, *, fallback: str = "none reported") -> str:
    if not isinstance(items, list) or not items:
        return fallback
    cleaned = [str(item).strip() for item in items if item not in (None, "")]
    if not cleaned:
        return fallback
    return "; ".join(cleaned)


def _format_match_item(item: Any) -> str:
    if not isinstance(item, dict):
        return _display_value(item)
    summary = _display_value(item.get("summary"), fallback="unnamed pattern")
    similarity = item.get("similarity")
    scope = _display_value(item.get("scope"), fallback="scope unavailable")
    if similarity is None:
        return f"{summary} ({scope})"
    return f"{summary} (similarity {similarity}, {scope})"


def _format_matches(matches: Any, *, fallback: str = "no comparable prior patterns listed") -> str:
    if not isinstance(matches, list) or not matches:
        return fallback
    rendered = [_format_match_item(item) for item in matches[:3]]
    return "; ".join(rendered)


def _doctrine_refusal_detail(reason: str | None) -> str:
    messages = {
        "prescriptive_or_actuation_language": "candidate language crossed into prescriptive/action phrasing",
        "insufficient_evidence": "evidence did not support a defensible doctrine assertion",
        "insufficient_corroboration": "corroborating signals were below doctrine minimum",
        "low_confidence": "confidence was below doctrine threshold",
    }
    return messages.get(reason or "", "candidate assertion did not pass doctrine checks")


def render_assistant_response(*, mode: AssistantMode, context: dict[str, Any]) -> dict[str, Any]:
    if mode not in ASSISTANT_MODES:
        raise ValueError(f"Unsupported assistant mode: {mode}")

    state = _as_dict(context.get("current_state"))
    risk = _as_dict(context.get("risk"))
    recommendation = _as_dict(context.get("recommendation"))
    memory = _as_dict(context.get("memory_recall"))
    novelty = _as_dict(memory.get("novelty"))
    nearest = _as_dict(memory.get("nearest_match"))
    recent_changes = _as_dict(context.get("recent_changes"))
    gate_decision = _as_dict(context.get("gate_decision"))

    observed: list[str] = [
        _grounded_text("Observed:", f"cycle={state.get('cycle')} timestamp={state.get('timestamp')}", "cycle/timestamp unavailable."),
        _grounded_text(
            "Observed:",
            f"risk_level={risk.get('risk_level')} trend={risk.get('trend')} latest_instability={risk.get('latest_instability')}",
            "risk assessment unavailable.",
        ),
        _grounded_text("Observed:", f"events={context.get('events')}", "event list unavailable."),
    ]

    inferred = [
        _grounded_text(
            "Inferred:",
            (
                f"novelty_is_novel={novelty.get('is_novel')} "
                f"nearest_match_found={nearest.get('found')} "
                f"nearest_similarity={nearest.get('similarity')}"
            ),
            "memory recall unavailable.",
        )
    ]

    candidate_action = str(recommendation.get("recommended_action") or "").strip()
    doctrine_eval = evaluate_statement(
        candidate_action,
        confidence=recommendation.get("recommendation_confidence"),
        corroborating_signals=count_confirming_signals(
            [
                bool(risk.get("risk_level")),
                bool(context.get("events")),
                bool(context.get("explanation")),
            ]
        ),
        doctrine=DEFAULT_DOCTRINE_V1,
    )

    doctrine_lines = [
        _grounded_text(
            "Doctrine:",
            f"status={'refused' if doctrine_eval.refused else 'allowed'} refusal_reason={doctrine_eval.refusal_reason}",
            "doctrine evaluation unavailable.",
        )
    ]
    if gate_decision:
        gate_line = (
            f"Gate decision={gate_decision.get('decision')} "
            f"reason={gate_decision.get('refusal_reason')} "
            f"confidence={gate_decision.get('confidence_label')}"
        )
        observed.append(_grounded_text("Observed:", gate_line, "gate decision unavailable."))

    if mode == "summary":
        text = "\n".join(["Current situation summary", *observed, *inferred, *doctrine_lines])
    elif mode == "why_recommended":
        text = "\n".join(
            [
                "Why this is being recommended",
                observed[1],
                _grounded_text("Observed:", context.get("explanation"), "explanation text unavailable."),
                doctrine_lines[0],
                "Inferred: Doctrine permits defensible state assertions only; it does not emit operator instructions.",
            ]
        )
    elif mode == "what_changed":
        text = "\n".join(
            [
                "What changed recently",
                _grounded_text("Observed:", f"risk_change={recent_changes.get('risk_level')}", "insufficient history."),
                _grounded_text("Observed:", f"new_events={recent_changes.get('new_events')}", "new events unavailable."),
                _grounded_text(
                    "Inferred:",
                    f"time_window={recent_changes.get('previous_timestamp')} -> {recent_changes.get('latest_timestamp')}",
                    "timeline window unavailable.",
                ),
                doctrine_lines[0],
            ]
        )
    elif mode == "pattern_similarity":
        text = "\n".join(
            [
                "Does this resemble a prior pattern",
                inferred[0],
                _grounded_text("Observed:", f"pattern_family={memory.get('pattern_family')}", "pattern family unavailable."),
                _grounded_text("Observed:", f"top_matches={memory.get('top_matches')}", "top matches unavailable."),
                "Doctrine: Treat pattern similarity as supporting evidence only.",
            ]
        )
    else:  # handoff
        text = "\n".join(
            [
                "Operator handoff note",
                _grounded_text(
                    "Observed:",
                    (
                        f"run_id={state.get('run_id')} site_id={state.get('site_id')} "
                        f"asset_id={state.get('asset_id')} cycle={state.get('cycle')}"
                    ),
                    "run/site/asset context unavailable.",
                ),
                observed[1],
                _grounded_text("Observed:", context.get("explanation"), "explanation unavailable."),
                inferred[0],
                doctrine_lines[0],
                "Doctrine: Handoff is state-only and avoids prescribed actions.",
            ]
        )

    return {
        "mode": mode,
        "text": text,
        "grounding": {
            "observed": observed,
            "inferred": inferred,
            "doctrine": doctrine_lines,
        },
        "context": context,
    }


def render_assistant_report(*, mode: ReportMode, context: dict[str, Any]) -> dict[str, Any]:
    if mode not in REPORT_MODES:
        raise ValueError(f"Unsupported report mode: {mode}")

    state = _as_dict(context.get("current_state"))
    risk = _as_dict(context.get("risk"))
    recommendation = _as_dict(context.get("recommendation"))
    memory = _as_dict(context.get("memory_recall"))
    nearest = _as_dict(memory.get("nearest_match"))
    novelty = _as_dict(memory.get("novelty"))
    pattern_family = _as_dict(memory.get("pattern_family"))
    recent_changes = _as_dict(context.get("recent_changes"))
    events = _as_list(context.get("events"))
    gate_decision = _as_dict(context.get("gate_decision"))

    rec_action = recommendation.get("recommended_action") or "No recommendation provided"
    rec_rationale = recommendation.get("rationale") or "No rationale provided"
    rec_confidence = recommendation.get("recommendation_confidence")
    operator_note = recommendation.get("operator_note") or "Operator note unavailable"
    supporting_evidence = recommendation.get("supporting_evidence")
    explanation = context.get("explanation") or "No explanation text available"

    confidence_text = (
        (
            f"Recommendation confidence is {_display_value(rec_confidence)}. "
            f"State confidence is {_display_value(state.get('confidence'))}."
        )
        if rec_confidence is not None or state.get("confidence") is not None
        else "Confidence values are not available."
    )
    nearest_found = bool(nearest.get("found"))
    nearest_similarity_text = (
        f" with similarity {_display_value(nearest.get('similarity'))}" if nearest_found else ""
    )

    shared_sections = {
        "Current System State": (
            "Run "
            f"{_display_value(state.get('run_id'))} at site {_display_value(state.get('site_id'))} for asset "
            f"{_display_value(state.get('asset_id'))}, cycle {_display_value(state.get('cycle'))}, timestamp "
            f"{_display_value(state.get('timestamp'))}."
        ),
        "Risk Assessment": (
            f"Risk level is {_display_value(risk.get('risk_level'))} with a "
            f"{_display_value(risk.get('trend'), fallback='unknown')} trend. "
            f"Latest instability score is {_display_value(risk.get('latest_instability'))}."
        ),
        "Recommended Next Step (advisory)": (
            f"Recommended advisory action: {rec_action}. "
            f"Rationale: {rec_rationale}. "
            f"Recommendation confidence: {_display_value(rec_confidence)}."
        ),
        "Supporting Evidence": (
            f"Explanation: {explanation} "
            f"Observed events: {_format_list(events)}. "
            f"Supporting evidence: {_display_value(supporting_evidence, fallback='none provided')}."
        ),
        "Pattern Context (memory recall if present)": (
            f"Novelty flag is {_display_value(novelty.get('is_novel'))}. "
            f"Nearest prior match was "
            f"{'found' if nearest_found else 'not found'}{nearest_similarity_text}. "
            f"Nearest match summary: {_display_value(nearest.get('summary'), fallback='not available')}. "
            f"Pattern family: {_display_value(pattern_family.get('label'), fallback='not available')} "
            f"(confidence {_display_value(pattern_family.get('confidence'))}). "
            f"Top matches: {_format_matches(memory.get('top_matches'))}."
        ),
        "Confidence": confidence_text,
        "Operator Note": str(operator_note),
    }
    if gate_decision:
        shared_sections["Aletheia Gate"] = (
            f"Decision: {_display_value(gate_decision.get('decision'))}. "
            f"Refusal reason: {_display_value(gate_decision.get('refusal_reason'))}. "
            f"Doctrine version: {_display_value(gate_decision.get('doctrine_version'))}. "
            f"Candidate assertion allowed: {_display_value(gate_decision.get('candidate_assertion_allowed'))}."
        )

    if mode == "client_report":
        sections = [
            (
                "Overview",
                "This advisory report summarizes the current operating condition and recommended next step "
                f"for cycle {_display_value(state.get('cycle'))} at {_display_value(state.get('timestamp'))}.",
            ),
            *shared_sections.items(),
        ]
        text = _join_sections("Client Report", sections)
    elif mode == "technician_summary":
        sections = [
            (
                "Current state (concise)",
                f"Risk is {_display_value(risk.get('risk_level'))} and trend is "
                f"{_display_value(risk.get('trend'), fallback='unknown')}. "
                f"Current advisory action: {rec_action}. Cycle {_display_value(state.get('cycle'))}.",
            ),
            (
                "What changed",
                f"Risk changed from {_display_value(_as_dict(recent_changes.get('risk_level')).get('previous'))} "
                f"to {_display_value(_as_dict(recent_changes.get('risk_level')).get('current'))}. "
                f"New events: {_format_list(recent_changes.get('new_events'))}. "
                f"Window: {_display_value(recent_changes.get('previous_timestamp'))} to "
                f"{_display_value(recent_changes.get('latest_timestamp'))}.",
            ),
            (
                "Key drivers",
                f"Explanation: {explanation} "
                f"Supporting evidence: {_display_value(supporting_evidence, fallback='none provided')}. "
                f"Events: {_format_list(events)}.",
            ),
            (
                "Recommended next step",
                f"Advisory action: {rec_action}. "
                f"Rationale: {rec_rationale}. "
                f"Recommendation confidence: {_display_value(rec_confidence)}. "
                f"Operator note: {_display_value(operator_note)}.",
            ),
        ]
        text = _join_sections("Technician Summary", sections)
    elif mode == "inspection_brief":
        sections = [
            (
                "Target system/component",
                f"Recommended target: {_display_value(recommendation.get('recommended_target'))}. "
                f"Advisory action: {rec_action}.",
            ),
            (
                "Why inspection is recommended",
                f"Rationale: {rec_rationale}. Explanation: {explanation}",
            ),
            (
                "What to check",
                f"Supporting evidence: {_display_value(supporting_evidence, fallback='none provided')}. "
                f"Observed events: {_format_list(events)}.",
            ),
            (
                "Risk context",
                f"Risk level {_display_value(risk.get('risk_level'))}, trend "
                f"{_display_value(risk.get('trend'), fallback='unknown')}, latest instability "
                f"{_display_value(risk.get('latest_instability'))}, recommendation confidence "
                f"{_display_value(rec_confidence)}.",
            ),
        ]
        text = _join_sections("Inspection Brief", sections)
    else:
        sections = [
            (
                "Overview",
                f"Handoff for run {_display_value(state.get('run_id'))}, cycle {_display_value(state.get('cycle'))}, "
                f"timestamp {_display_value(state.get('timestamp'))}.",
            ),
            ("Current System State", shared_sections["Current System State"]),
            ("Risk Assessment", shared_sections["Risk Assessment"]),
            ("Recommended Next Step (advisory)", shared_sections["Recommended Next Step (advisory)"]),
            ("Supporting Evidence", shared_sections["Supporting Evidence"]),
            ("Pattern Context (memory recall if present)", shared_sections["Pattern Context (memory recall if present)"]),
            ("Confidence", shared_sections["Confidence"]),
            ("Operator Note", shared_sections["Operator Note"]),
        ]
        text = _join_sections("Handoff Note", sections)

    sections_dict = {name: content for name, content in sections}
    return {
        "mode": mode,
        "report_text": text,
        "sections": sections_dict,
        "context": context,
    }
