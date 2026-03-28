from __future__ import annotations

from typing import Any, Literal

ASSISTANT_MODES = {
    "summary",
    "why_recommended",
    "what_changed",
    "pattern_similarity",
    "handoff",
}

AssistantMode = Literal["summary", "why_recommended", "what_changed", "pattern_similarity", "handoff"]


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

    recommended = [
        _grounded_text(
            "Recommended:",
            (
                f"action={recommendation.get('recommended_action')} "
                f"confidence={recommendation.get('recommendation_confidence')} "
                f"rationale={recommendation.get('rationale')}"
            ),
            "no recommendation provided.",
        ),
        _grounded_text("Recommended:", recommendation.get("operator_note"), "operator safety note unavailable."),
    ]

    if mode == "summary":
        text = "\n".join(["Current situation summary", *observed, *inferred, *recommended])
    elif mode == "why_recommended":
        text = "\n".join(
            [
                "Why this is being recommended",
                observed[1],
                _grounded_text("Observed:", context.get("explanation"), "explanation text unavailable."),
                recommended[0],
                "Inferred: Recommendation rationale is treated as advisory, not autonomous instruction.",
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
                recommended[0],
            ]
        )
    elif mode == "pattern_similarity":
        text = "\n".join(
            [
                "Does this resemble a prior pattern",
                inferred[0],
                _grounded_text("Observed:", f"pattern_family={memory.get('pattern_family')}", "pattern family unavailable."),
                _grounded_text("Observed:", f"top_matches={memory.get('top_matches')}", "top matches unavailable."),
                "Recommended: Treat pattern similarity as supporting evidence only.",
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
                recommended[0],
                "Recommended: Continue with site SOP verification before acting.",
            ]
        )

    return {
        "mode": mode,
        "text": text,
        "grounding": {
            "observed": observed,
            "inferred": inferred,
            "recommended": recommended,
        },
        "context": context,
    }
