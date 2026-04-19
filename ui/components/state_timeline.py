"""State Timeline: Bottom-integrated state evolution visualization.

Shows the progression of system states over time:
- Baseline: Normal operation
- Drift: Rising structural drift but still stable
- Emerging: Instability beginning to form
- Persistent: Instability sustained over time
- Accelerated: Rapid change detected
- Critical: System failure threshold approaching

Visual language shared with system field:
- Smooth curves during stable periods
- Deformation shows drift progression
- Colors match system field palette
- Hover to preview past system states
- Subtle projected trajectory showing where system is heading
"""

from __future__ import annotations

from typing import Any
from html import escape


def render_state_timeline(
    states: list[dict[str, Any]],
    current_index: int | None = None,
    width: int = 1200,
    height: int = 140,
) -> str:
    """Render the state timeline.

    Args:
        states: List of state dicts with keys:
            timestamp, state_label, coherence, drift, stability,
            is_admitted, emphasis
        current_index: Index of current position in timeline
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        SVG markup for the timeline
    """

    if not states:
        states = [
            {
                "timestamp": "—",
                "state_label": "No Data",
                "coherence": 0.5,
                "drift": 0.0,
                "stability": 0.5,
                "is_admitted": False,
                "emphasis": "low",
            }
        ]

    state_colors = {
        "baseline": "#6B7280",
        "drift": "#3B82F6",
        "emerging": "#F97316",
        "persistent": "#EF4444",
        "accelerated": "#DC2626",
        "critical": "#991B1B",
    }

    margin_left = 40
    margin_right = 40
    margin_top = 20
    margin_bottom = 40

    available_width = width - margin_left - margin_right
    available_height = height - margin_top - margin_bottom

    if len(states) < 2:
        x_spacing = available_width
    else:
        x_spacing = available_width / (len(states) - 1)

    svg_parts = []

    svg_parts.append(f"""<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"
        xmlns="http://www.w3.org/2000/svg" class="ner-state-timeline">
        <defs>
            <filter id="timelineGlow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge>
                    <feMergeNode in="blur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <linearGradient id="timelineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#6B7280"/>
                <stop offset="50%" stop-color="#3B82F6"/>
                <stop offset="100%" stop-color="#F97316"/>
            </linearGradient>
        </defs>
    """)

    svg_parts.append(f"""
        <rect width="{width}" height="{height}" fill="#020617" opacity="0"/>
    """)

    baseline_y = margin_top + available_height * 0.5

    points_x = []
    points_coherence = []

    for i, state in enumerate(states):
        x = margin_left + (i * x_spacing)
        points_x.append(x)

        coherence = float(state.get("coherence", 0.5))
        coherence_y = baseline_y - (coherence - 0.5) * available_height * 0.8
        points_coherence.append((x, coherence_y))

    if len(points_coherence) > 1:
        line_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in points_coherence)
        svg_parts.append(f"""
            <polyline points="{line_points}" fill="none" stroke="url(#timelineGradient)"
                stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        """)

    for i, state in enumerate(states):
        x = points_x[i]
        y = points_coherence[i][1] if i < len(points_coherence) else baseline_y

        state_label = str(state.get("state_label", "unknown")).lower()
        state_color = state_colors.get(state_label, "#94A3B8")

        is_admitted = bool(state.get("is_admitted", False))
        emphasis = str(state.get("emphasis", "low")).lower()

        node_radius = 5.0 if emphasis == "low" else 7.0 if emphasis == "medium" else 9.0

        if is_admitted:
            svg_parts.append(f"""
                <circle cx="{x:.1f}" cy="{y:.1f}" r="{node_radius + 2.5}"
                    fill="{state_color}" opacity="0.3" filter="url(#timelineGlow)"/>
            """)

        svg_parts.append(f"""
            <circle cx="{x:.1f}" cy="{y:.1f}" r="{node_radius}"
                fill="{state_color}" opacity="0.9" stroke="#E2E8F0" stroke-width="1.5"/>
        """)

        if i == current_index:
            svg_parts.append(f"""
                <circle cx="{x:.1f}" cy="{y:.1f}" r="{node_radius + 4}"
                    fill="none" stroke="#22D3EE" stroke-width="2" opacity="0.8"
                    stroke-dasharray="4,2"/>
            """)

        timestamp = str(state.get("timestamp", "")).split("T")[-1].split("Z")[0] if state.get("timestamp") else "—"
        svg_parts.append(f"""
            <text x="{x:.1f}" y="{margin_top + available_height + 20}" text-anchor="middle"
                font-size="10" fill="rgba(148,163,184,0.8)">{escape(timestamp)}</text>
        """)

    legend_y = margin_top + available_height + 35
    legend_items = [
        ("Baseline", "#6B7280"),
        ("Drift", "#3B82F6"),
        ("Emerging", "#F97316"),
        ("Critical", "#DC2626"),
    ]

    legend_x = margin_left
    for label, color in legend_items:
        svg_parts.append(f"""
            <circle cx="{legend_x}" cy="{legend_y}" r="3"
                fill="{color}"/>
        """)
        svg_parts.append(f"""
            <text x="{legend_x + 12}" y="{legend_y + 3}" font-size="9"
                fill="rgba(226,232,240,0.7)">{escape(label)}</text>
        """)
        legend_x += 110

    svg_parts.append("</svg>")

    return "\n".join(svg_parts)
