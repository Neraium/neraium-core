from __future__ import annotations

from ui.components import (
    render_alert_fsm_widget,
    render_attribution_viz,
    render_causal_inspector,
    render_mode_selector,
    render_regime_timeline,
    render_structural_flow_viz,
)
from ui.core_integration import SystemState


def build_navigation_surface(state: SystemState, *, mode: str, width_px: int) -> dict[str, object]:
    viewport = "mobile" if width_px < 760 else "desktop"
    return {
        "surface": render_structural_flow_viz(state),
        "overlays": {
            "mode": render_mode_selector(mode),
            "timeline": render_regime_timeline(state),
            "relationships": render_causal_inspector(state),
            "attribution": render_attribution_viz(state),
            "phase": render_alert_fsm_widget(state),
        },
        "viewport": viewport,
    }
