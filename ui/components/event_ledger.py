from __future__ import annotations

from typing import Any


def render_event_ledger(recent_events: list[dict[str, Any]] | None) -> dict[str, Any]:
    events = recent_events or []
    compact = [
        {
            "timestamp": event.get("timestamp"),
            "gate_decision": event.get("decision") or ("ADMIT" if event.get("event_admitted") else "SUPPRESS"),
            "doctrine_version": event.get("doctrine_version") or "unknown",
            "summary": event.get("evidence_summary") or "No evidence summary is available.",
            "previous_state_summary": event.get("previous_state_summary") or "No previous state summary.",
            "current_state_summary": event.get("current_state_summary") or "No current state summary.",
            "transition_type": event.get("transition_type") or "STABLE",
            "hash": event.get("hash") or "hash:pending",
        }
        for event in events
    ]
    return {
        "component": "evidence_record",
        "title": "Evidence Record",
        "entries": compact,
    }
