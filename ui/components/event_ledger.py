from __future__ import annotations

from typing import Any


def render_event_ledger(recent_events: list[dict[str, Any]] | None) -> dict[str, Any]:
    events = recent_events or []
    compact = [
        {
            "timestamp": event.get("timestamp"),
            "event_summary": event.get("evidence_summary") or "No event summary is available.",
            "decision": event.get("decision") or ("ADMIT" if event.get("event_admitted") else None),
            "doctrine_version": event.get("doctrine_version"),
        }
        for event in events
    ]
    return {
        "component": "event_ledger",
        "title": "Recent Events / Record",
        "events": compact,
    }
