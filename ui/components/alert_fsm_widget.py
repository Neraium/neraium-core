from __future__ import annotations

from ui.core_integration import SystemState


def render_alert_fsm_widget(state: SystemState) -> dict[str, object]:
    if state.drift_intensity >= 0.7:
        phase = "divergence"
    elif state.drift_intensity >= 0.4:
        phase = "transition"
    else:
        phase = "coherence"
    return {
        "overlay": "regime_phase_badge",
        "phase": phase,
        "regime": state.regime_state,
        "drift": round(state.drift_intensity, 4),
    }
