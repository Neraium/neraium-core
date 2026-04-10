from __future__ import annotations

from ui.core_integration import SystemState


def render_attribution_viz(state: SystemState) -> dict[str, object]:
    """Contextual overlay explaining why the position is moving."""
    factors = [
        {"factor": "drift_intensity", "value": round(state.drift_intensity, 4)},
        {"factor": "trajectory_speed", "value": round(min(1.0, abs(state.velocity[0]) + abs(state.velocity[1])), 4)},
        {"factor": "confidence", "value": round(state.confidence, 4)},
    ]
    return {"overlay": "context_attribution", "factors": sorted(factors, key=lambda x: x["value"], reverse=True)}
