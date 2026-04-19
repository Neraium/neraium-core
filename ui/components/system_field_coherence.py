"""System Field with Coherence Core.

The heart of the Neraium operational interface:
- Dynamic tetrahedral structure representing subsystem relationships
- Coherence Core at center that pulses, glows, and degrades based on system health
- Visual deformation reflects system state: stable, drift, instability, critical
- Interactive: hover/click subsystems to focus and explore

The tetrahedron represents:
  Vertices: Climate, Airflow, Irrigation, Plant Response (or generic domains)
  Edges: Relationships and coupling between subsystems
  Center: System Coherence (unity, integration, health)
"""

from __future__ import annotations

import math
from typing import Any

from ui.utils import clamp, safe_float


SUBSYSTEM_DOMAINS = {
    "climate": {"label": "Climate", "angle": 0},
    "airflow": {"label": "Airflow", "angle": 90},
    "irrigation": {"label": "Irrigation", "angle": 180},
    "plant_response": {"label": "Plant Response", "angle": 270},
}


def _compute_tetrahedron_geometry(
    coherence: float,
    drift: float,
    stability: float,
    state: str = "stable",
) -> dict[str, Any]:
    """Compute tetrahedron vertex positions based on system state.

    Args:
        coherence: Coherence score (0-1). Controls core glow and ring stability.
        drift: Structural drift (0-1). Controls deformation and tension.
        stability: Relational stability (0-1). Controls edge smoothness.
        state: One of stable, drift, instability, critical

    Returns:
        Dictionary with vertex positions, edge states, coherence core params
    """

    base_scale = 1.0

    if state == "stable":
        deformation = 0.0
        tension_color = "#22C55E"
        edge_opacity = 0.8
        core_radius = 0.15
        core_pulse_intensity = 0.3
    elif state == "drift":
        deformation = drift * 0.15
        tension_color = "#3B82F6"
        edge_opacity = 0.6 + 0.2 * drift
        core_radius = 0.12 + 0.03 * drift
        core_pulse_intensity = 0.4 + 0.2 * drift
    elif state == "instability":
        deformation = drift * 0.25
        tension_color = "#F97316"
        edge_opacity = 0.7 + 0.2 * drift
        core_radius = 0.1 + 0.05 * drift
        core_pulse_intensity = 0.5 + 0.3 * drift
    else:  # critical
        deformation = 0.35
        tension_color = "#EF4444"
        edge_opacity = 0.9
        core_radius = 0.08 + 0.07 * (1.0 - coherence)
        core_pulse_intensity = 0.6 + 0.4 * (1.0 - coherence)

    base_vertices = {
        "top": (0, -0.5 * base_scale, 0.5 * base_scale),
        "front_right": (0.5 * base_scale, 0.3 * base_scale, 0.3 * base_scale),
        "front_left": (-0.5 * base_scale, 0.3 * base_scale, 0.3 * base_scale),
        "back": (0, 0.3 * base_scale, -0.5 * base_scale),
    }

    asymmetry_factor = clamp(deformation, 0.0, 0.4)
    drift_direction = (
        math.sin(drift * math.pi),
        0,
        math.cos(drift * math.pi * 0.5),
    )

    deformed_vertices = {}
    for name, (x, y, z) in base_vertices.items():
        dx = drift_direction[0] * asymmetry_factor * abs(x)
        dy = drift_direction[1] * asymmetry_factor * abs(y)
        dz = drift_direction[2] * asymmetry_factor * abs(z)

        scale_factor = 1.0 - (asymmetry_factor * 0.2) if asymmetry_factor > 0 else 1.0

        deformed_vertices[name] = (
            (x + dx) * scale_factor,
            (y + dy) * scale_factor,
            (z + dz) * scale_factor,
        )

    edges = [
        ("top", "front_right"),
        ("top", "front_left"),
        ("top", "back"),
        ("front_right", "front_left"),
        ("front_right", "back"),
        ("front_left", "back"),
    ]

    edge_states = []
    for v1_name, v2_name in edges:
        v1 = deformed_vertices[v1_name]
        v2 = deformed_vertices[v2_name]

        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
        stress_factor = clamp(abs(dist - 1.0), 0.0, 1.0)

        edge_states.append({
            "from": v1_name,
            "to": v2_name,
            "p1": v1,
            "p2": v2,
            "tension": stress_factor,
            "glow_intensity": (stress_factor * 0.8) + (edge_opacity * 0.2),
            "color": tension_color,
        })

    coherence_ring = {
        "radius": 0.65,
        "is_stable": state == "stable",
        "deformation": clamp(deformation, 0.0, 0.3),
        "glow_color": tension_color,
        "glow_intensity": 0.4 + (1.0 - coherence) * 0.6,
        "opacity": 0.5 + (coherence * 0.5),
    }

    coherence_core = {
        "radius": core_radius,
        "color": "#22C55E" if coherence > 0.6 else "#3B82F6" if coherence > 0.3 else "#F97316",
        "pulse_intensity": core_pulse_intensity,
        "pulse_frequency": 1.0 + (drift * 0.5),
        "glow_spread": 0.15 + (drift * 0.1),
        "opacity": 0.8 + (coherence * 0.2),
    }

    subsystem_influence = []
    for domain_key, domain_info in SUBSYSTEM_DOMAINS.items():
        angle = math.radians(domain_info["angle"])
        influence_distance = 0.75
        x = influence_distance * math.cos(angle)
        y = influence_distance * math.sin(angle)

        magnitude = clamp(drift * (1.0 if "climate" in domain_key else 0.7), 0.0, 1.0)

        subsystem_influence.append({
            "domain": domain_key,
            "label": domain_info["label"],
            "position": (x, y),
            "influence_magnitude": magnitude,
            "influence_direction": (math.cos(angle), math.sin(angle)),
        })

    return {
        "vertices": deformed_vertices,
        "edges": edge_states,
        "coherence_ring": coherence_ring,
        "coherence_core": coherence_core,
        "subsystem_influence": subsystem_influence,
        "state": state,
        "drift": drift,
        "stability": stability,
        "coherence": coherence,
    }


def render_system_field_svg(
    coherence: float,
    drift: float,
    stability: float,
    state: str = "stable",
    width: int = 800,
    height: int = 600,
    interactive: bool = True,
) -> str:
    """Render the system field as an interactive SVG.

    Args:
        coherence: System coherence score (0-1)
        drift: Structural drift score (0-1)
        stability: Relational stability score (0-1)
        state: System state (stable, drift, instability, critical)
        width: SVG width in pixels
        height: SVG height in pixels
        interactive: If True, add hover/click interactivity

    Returns:
        SVG markup string
    """
    geometry = _compute_tetrahedron_geometry(coherence, drift, stability, state)

    center_x = width / 2
    center_y = height / 2
    scale = min(width, height) * 0.25

    def to_svg(x: float, y: float, z: float) -> tuple[float, float]:
        """Project 3D tetrahedron to 2D SVG.

        Simple isometric projection for clarity and performance.
        """
        proj_x = (x - y) * 0.866
        proj_y = (x + y) * 0.5 - z
        return center_x + proj_x * scale, center_y + proj_y * scale

    svg_parts = []

    svg_parts.append(f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
        xmlns="http://www.w3.org/2000/svg" class="ner-system-field">
        <defs>
            <filter id="coherenceGlow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="8" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <filter id="coreRadiance" x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur stdDeviation="12" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <style>
                @keyframes coherencePulse {{
                    0%, 100% {{ opacity: 0.6; }}
                    50% {{ opacity: 1.0; }}
                }}
                @keyframes coreRadiate {{
                    0% {{ r: {geometry["coherence_core"]["radius"]:.3f}; }}
                    50% {{ r: {geometry["coherence_core"]["radius"] + 0.05:.3f}; }}
                    100% {{ r: {geometry["coherence_core"]["radius"]:.3f}; }}
                }}
                .ner-coherence-core {{
                    animation: coreRadiate {1.0 / geometry["coherence_core"]["pulse_frequency"]:.2f}s ease-in-out infinite;
                }}
            </style>
        </defs>
    """)

    svg_parts.append(f"""
        <rect width="{width}" height="{height}" fill="#020617" opacity="0"/>
    """)

    coherence_ring = geometry["coherence_ring"]
    ring_cx, ring_cy = center_x, center_y
    ring_r = coherence_ring["radius"] * scale

    glow_opacity = coherence_ring["glow_intensity"] * 0.4
    svg_parts.append(f"""
        <circle cx="{ring_cx}" cy="{ring_cy}" r="{ring_r}"
            fill="none" stroke="{coherence_ring["glow_color"]}"
            stroke-width="2.5" opacity="{glow_opacity}" filter="url(#coherenceGlow)"/>
    """)

    svg_parts.append(f"""
        <circle cx="{ring_cx}" cy="{ring_cy}" r="{ring_r}"
            fill="none" stroke="{coherence_ring["glow_color"]}"
            stroke-width="1.2" opacity="{coherence_ring["opacity"]}"/>
    """)

    for edge in geometry["edges"]:
        x1, y1 = to_svg(*edge["p1"])
        x2, y2 = to_svg(*edge["p2"])

        edge_color = edge["color"]
        edge_width = 1.2 + edge["tension"] * 2.0
        edge_opacity = edge["glow_intensity"]

        svg_parts.append(f"""
            <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"
                stroke="{edge_color}" stroke-width="{edge_width:.2f}"
                opacity="{edge_opacity}" stroke-linecap="round"/>
        """)

    vertices = geometry["vertices"]
    for v_name, (vx, vy, vz) in vertices.items():
        vx_svg, vy_svg = to_svg(vx, vy, vz)

        is_highlighted = False
        v_color = "#93C5FD"
        v_radius = 6.0
        v_opacity = 0.8

        svg_parts.append(f"""
            <circle cx="{vx_svg:.1f}" cy="{vy_svg:.1f}" r="{v_radius}"
                fill="{v_color}" opacity="{v_opacity}"/>
        """)

    core = geometry["coherence_core"]
    core_x, core_y = center_x, center_y
    core_r = core["radius"] * scale

    core_glow_r = core_r + core["glow_spread"] * scale
    svg_parts.append(f"""
        <circle cx="{core_x}" cy="{core_y}" r="{core_glow_r}"
            fill="{core["color"]}" opacity="{core["pulse_intensity"] * 0.3}"
            filter="url(#coreRadiance)"/>
    """)

    svg_parts.append(f"""
        <circle cx="{core_x}" cy="{core_y}" r="{core_r}"
            fill="{core["color"]}" opacity="{core["opacity"]}"
            class="ner-coherence-core" filter="url(#coreRadiance)"/>
    """)

    if interactive:
        for subsys in geometry["subsystem_influence"]:
            sx, sy = subsys["position"]
            sx_svg = center_x + sx * scale
            sy_svg = center_y + sy * scale

            mag = subsys["influence_magnitude"]
            mag_radius = 8 + mag * 6
            mag_opacity = 0.3 + mag * 0.4

            svg_parts.append(f"""
                <circle cx="{sx_svg:.1f}" cy="{sy_svg:.1f}" r="{mag_radius:.1f}"
                    fill="#3B82F6" opacity="{mag_opacity}"
                    class="ner-subsystem-vertex" data-subsystem="{subsys["domain"]}"/>
            """)

            if mag > 0.1:
                dir_x, dir_y = subsys["influence_direction"]
                arrow_end_x = sx_svg + dir_x * mag * scale * 0.3
                arrow_end_y = sy_svg + dir_y * mag * scale * 0.3

                svg_parts.append(f"""
                    <line x1="{sx_svg:.1f}" y1="{sy_svg:.1f}"
                        x2="{arrow_end_x:.1f}" y2="{arrow_end_y:.1f}"
                        stroke="#3B82F6" stroke-width="1.5"
                        opacity="{0.4 + mag * 0.4}" stroke-linecap="round"/>
                """)

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)
