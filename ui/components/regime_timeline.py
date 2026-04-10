from __future__ import annotations

from ui.core_integration import SystemState


def render_regime_timeline(state: SystemState) -> dict[str, object]:
    """Bottom-strip overlay for structural evolution events."""
    return {
        "overlay": "timeline_strip",
        "events": [
            {
                "timestamp": event.t,
                "regime": event.regime,
                "drift_acceleration": event.drift_delta,
                "reaction_window_minutes": event.reaction_window_minutes,
            }
            for event in state.timeline
        ],
    }
