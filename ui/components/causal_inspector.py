from __future__ import annotations

from ui.core_integration import SystemState


def render_causal_inspector(state: SystemState) -> dict[str, object]:
    return {
        "overlay": "structural_explanation",
        "state": {
            "regime_name": state.regime_state,
            "system_health": state.system_health,
            "structural_drift_score": state.drift_intensity,
            "confidence": state.confidence,
        },
        "relationship_changes": [
            {
                "edge": f"{edge.source}->{edge.target}",
                "strength": edge.strength,
                "trend": edge.trend,
            }
            for edge in state.structural_relationships
        ],
    }
