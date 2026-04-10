from __future__ import annotations

from ui.core_integration import SystemState
from ui.utils import clamp, l2_norm


def _with_alpha(hex_color: str, alpha: float) -> str:
    """Render color as rgba string for soft trail gradients."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(125, 142, 255, {clamp(alpha, 0.0, 1.0):.3f})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {clamp(alpha, 0.0, 1.0):.3f})"


def render_structural_flow_viz(state: SystemState) -> dict[str, object]:
    """Trajectory-first spatial navigation field, not a chart."""
    tail = [{"x": p.x, "y": p.y, "t": p.t} for p in state.trajectory_history]
    velocity_mag = l2_norm(state.velocity)
    trail_count = max(len(tail), 1)

    fading_tail = [
        {
            **point,
            "opacity": round(0.12 + (idx + 1) / trail_count * 0.82, 4),
            "glow": round(0.12 + (idx + 1) / trail_count * 0.68, 4),
            "radius": round(2.0 + (idx + 1) / trail_count * 4.5, 3),
            "color": _with_alpha("#6E8DFF", 0.08 + (idx + 1) / trail_count * 0.48),
        }
        for idx, point in enumerate(tail)
    ]

    vector_tip = {
        "x": clamp(state.position.x + state.velocity[0] * 2.4, 0.0, 1.0),
        "y": clamp(state.position.y + state.velocity[1] * 2.4, 0.0, 1.0),
    }
    regime_phase = "coherence"
    if state.drift_intensity >= 0.7:
        regime_phase = "divergence"
    elif state.drift_intensity >= 0.4:
        regime_phase = "transition"

    return {
        "surface": "system_navigation_field",
        "canvas": {
            "style": "dark_spatial",
            "frame": "frameless",
            "axes": "hidden",
            "grid": "off",
            "chrome": "none",
            "background": {
                "base": "#050812",
                "radial_tint": ["#101532", "#080f25", "#05070E"],
                "vignette": 0.42,
                "noise": 0.03,
            },
        },
        "trajectory": {
            "type": "continuous_tail",
            "path": tail,
            "fading_tail": fading_tail,
            "gradient_trail": {
                "from": _with_alpha("#67D5FF", 0.2),
                "via": _with_alpha("#7B89FF", 0.36),
                "to": _with_alpha("#A85DFF", 0.6),
            },
            "interpolation": {
                "enabled": True,
                "style": "catmull_rom",
                "substeps": 3,
                "smoothing": round(clamp(0.45 + velocity_mag, 0.45, 0.92), 3),
            },
        },
        "current_position": {
            "x": state.position.x,
            "y": state.position.y,
            "t": state.position.t,
            "highlight": "focus_node",
            "halo_radius": round(clamp(0.02 + velocity_mag * 0.1, 0.02, 0.12), 4),
            "glow": round(clamp(0.38 + state.drift_intensity * 0.42, 0.38, 0.96), 3),
            "phase": regime_phase,
        },
        "velocity_vector": {
            "origin": {"x": state.position.x, "y": state.position.y},
            "tip": vector_tip,
            "dx": state.velocity[0],
            "dy": state.velocity[1],
            "magnitude": round(velocity_mag, 6),
            "arrow": {"style": "soft", "glow": 0.62, "head_size": "small"},
        },
        "projected_forward_region": {
            "kind": "cone",
            "samples": [{"x": p.x, "y": p.y, "t": p.t} for p in state.projected_cone],
            "opacity": round(clamp(0.15 + velocity_mag * 1.4, 0.15, 0.7), 3),
            "blur": round(clamp(0.35 + state.drift_intensity * 0.4, 0.35, 0.82), 3),
        },
        "stability_region_hints": {
            "bands": state.stability_regions,
            "stable_tone": _with_alpha("#6FD6FF", 0.18),
            "transition_tone": _with_alpha("#8D7DFF", 0.16),
            "divergence_tone": _with_alpha("#FF9B7D", 0.2),
        },
    }
