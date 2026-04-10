from __future__ import annotations

from ui.components import (
    render_alert_fsm_widget,
    render_attribution_viz,
    render_causal_inspector,
    render_mode_selector,
    render_reaction_window_indicator,
    render_regime_timeline,
    render_structural_flow_viz,
)
from ui.core_integration import SystemState


MODE_OVERLAYS: dict[str, tuple[str, ...]] = {
    "pilot": ("mode", "timeline", "phase", "reaction_window"),
    "operations": ("mode", "timeline", "phase", "reaction_window", "relationships"),
    "demo": ("mode", "timeline", "phase", "reaction_window", "attribution", "relationships"),
}


def build_navigation_surface(state: SystemState, *, mode: str, width_px: int) -> dict[str, object]:
    viewport = "mobile" if width_px < 760 else "desktop"
    resolved_mode = mode if mode in MODE_OVERLAYS else "pilot"

    overlay_catalog = {
        "mode": render_mode_selector(resolved_mode),
        "timeline": render_regime_timeline(state),
        "relationships": render_causal_inspector(state),
        "attribution": render_attribution_viz(state),
        "phase": render_alert_fsm_widget(state),
        "reaction_window": render_reaction_window_indicator(state),
    }

    return {
        "surface": render_structural_flow_viz(state),
        "overlays": {key: overlay_catalog[key] for key in MODE_OVERLAYS[resolved_mode]},
        "viewport": viewport,
        "composition": {
            "layout": "single_surface",
            "framing": "open",
            "chrome": "low_contrast",
        },
    }
