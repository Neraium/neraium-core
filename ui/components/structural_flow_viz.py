from __future__ import annotations

from ui.core_integration import SystemState


def render_structural_flow_viz(state: SystemState) -> dict[str, object]:
    """Primary navigation surface: full system trajectory map."""
    return {
        "surface": "system_navigation_surface",
        "position": {"x": state.position.x, "y": state.position.y, "t": state.position.t},
        "trajectory_tail": [{"x": p.x, "y": p.y, "t": p.t} for p in state.trajectory_history],
        "velocity_vector": {"dx": state.velocity[0], "dy": state.velocity[1]},
        "projected_cone": [{"x": p.x, "y": p.y, "t": p.t} for p in state.projected_cone],
        "stability_regions": state.stability_regions,
    }
