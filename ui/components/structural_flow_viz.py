from __future__ import annotations

from typing import Any

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


def _event_point_style(
    point: dict[str, Any],
    *,
    event_type: str,
    recency_ratio: float,
) -> dict[str, Any]:
    """Apply event-specific visual treatment while keeping path geometry unchanged."""
    if event_type == "admitted":
        return {
            **point,
            "event_type": "admitted",
            "color": _with_alpha("#62FFB3", 0.55 + recency_ratio * 0.35),
            "glow": round(0.48 + recency_ratio * 0.42, 3),
            "opacity": round(0.74 + recency_ratio * 0.22, 3),
            "radius": round(3.2 + recency_ratio * 3.1, 3),
            "blend_mode": "screen",
        }
    return {
        **point,
        "event_type": "suppressed",
        "color": _with_alpha("#FF6B8A", 0.3 + recency_ratio * 0.3),
        "glow": round(0.16 + recency_ratio * 0.2, 3),
        "opacity": round(0.28 + recency_ratio * 0.3, 3),
        "radius": round(2.7 + recency_ratio * 2.1, 3),
        "blend_mode": "multiply",
    }


def render_structural_flow_viz(
    state: SystemState,
    gate_decision: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
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

    decision = (gate_decision or {}).get("decision")
    admit_points = [point for point in tail if decision == "ADMIT"]
    suppress_points = [point for point in tail if decision == "SUPPRESS"]
    void_points = [point for point in tail if decision == "ADMISSIBILITY_VOID"]
    rows = records or []
    indexed_tail = list(enumerate(tail))

    stable_points = [point for idx, point in indexed_tail if idx < len(rows) and str(rows[idx].get("transition_type", "STABLE")).upper() == "STABLE"]
    transition_points = [
        point for idx, point in indexed_tail if idx < len(rows) and str(rows[idx].get("transition_type", "")).upper() == "TRANSITION"
    ]
    reorganization_points = [
        point for idx, point in indexed_tail if idx < len(rows) and str(rows[idx].get("transition_type", "")).upper() == "REORGANIZATION"
    ]
    admitted_event_points = [point for idx, point in indexed_tail if idx < len(rows) and bool(rows[idx].get("event_admitted"))]
    if str(decision or "SUPPRESS").upper() == "SUPPRESS" and rows:
        admitted_event_points = [point for point in admitted_event_points if point.get("t") != rows[-1].get("timestamp")]
    suppressed_event_points = [
        point for idx, point in indexed_tail if idx < len(rows) and rows[idx].get("event_admitted") is False
    ]
    admitted_styled_points = [
        _event_point_style(point, event_type="admitted", recency_ratio=(idx + 1) / max(len(admitted_event_points), 1))
        for idx, point in enumerate(admitted_event_points)
    ]
    suppressed_styled_points = [
        _event_point_style(point, event_type="suppressed", recency_ratio=(idx + 1) / max(len(suppressed_event_points), 1))
        for idx, point in enumerate(suppressed_event_points)
    ]

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
        "gate_coupling": {
            "decision": decision,
            "admit_highlights": {
                "points": admit_points,
                "style": "highlight",
                "color": _with_alpha("#62FFB3", 0.85),
            },
            "suppress_regions": {
                "points": suppress_points,
                "style": "fade",
                "color": _with_alpha("#FF8B8B", 0.55),
            },
            "void_regions": {
                "points": void_points,
                "style": "uncertain",
                "color": _with_alpha("#BBA1FF", 0.65),
            },
        },
        "phase_layers": {
            "stable_baseline": stable_points,
            "transition": transition_points,
            "reorganization": reorganization_points,
            "admitted_events": admitted_styled_points,
            "suppressed_events": suppressed_styled_points,
            "event_visual_encoding": {
                "continuity_anchor": "trajectory.path",
                "admitted": {
                    "color": "#62FFB3",
                    "glow_range": [0.48, 0.9],
                    "opacity_range": [0.74, 0.96],
                },
                "suppressed": {
                    "color": "#FF6B8A",
                    "glow_range": [0.16, 0.36],
                    "opacity_range": [0.28, 0.58],
                },
            },
        },
    }
