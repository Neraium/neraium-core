"""System Geometry Visualization: Structural representation of system relationships.

Replaces the line chart with a geometric view showing:
- Nodes: Sensors or functional units
- Edges: Relationships/coupling between nodes
- Structure deformation: Reflects stability vs. drift
- Trails: Prior stable states leave subtle geometric markers

Visual encoding:
- Tight, clustered structure = Stable
- Deformed structure = Drift
- Transitioning structure = Active change
- Settled structure = New equilibrium
"""

from __future__ import annotations

import math
from typing import Any

from ui.utils import clamp


def _generate_sensor_nodes(num_sensors: int = 8) -> list[dict[str, Any]]:
    """Generate a regular arrangement of sensor nodes in a circle.

    Args:
        num_sensors: Number of sensors (nodes) to arrange

    Returns:
        List of node positions in normalized [0, 1] space
    """
    nodes = []
    for i in range(num_sensors):
        angle = (i / num_sensors) * 2 * math.pi
        # Arrange in circle with center offset to create visual interest
        x = 0.5 + 0.35 * math.cos(angle)
        y = 0.5 + 0.35 * math.sin(angle)
        nodes.append({
            "id": f"sensor_{i}",
            "index": i,
            "x": clamp(x, 0.0, 1.0),
            "y": clamp(y, 0.0, 1.0),
            "label": f"S{i+1}",
            "base_x": x,
            "base_y": y,
        })
    return nodes


def _apply_drift_deformation(nodes: list[dict[str, Any]], drift_intensity: float, velocity: list[float]) -> list[dict[str, Any]]:
    """Apply drift-based deformation to node positions.

    Nodes shift in the direction of system velocity, with magnitude proportional
    to drift intensity. Creates visual sense of structure being "pulled" or deformed.

    Args:
        nodes: Base node positions
        drift_intensity: Magnitude of drift (0-1)
        velocity: System velocity vector [vx, vy]

    Returns:
        Deformed node positions
    """
    deformed = []
    for node in nodes:
        # Distance from center affects deformation magnitude
        dx_from_center = node["base_x"] - 0.5
        dy_from_center = node["base_y"] - 0.5
        distance_from_center = math.sqrt(dx_from_center**2 + dy_from_center**2)

        # Outer nodes deform more (closer to perimeter)
        deformation_factor = (distance_from_center / 0.35) if distance_from_center > 0 else 0.0

        # Apply drift-based displacement
        vx, vy = velocity[0] if len(velocity) > 0 else 0.0, velocity[1] if len(velocity) > 1 else 0.0
        vx = float(vx or 0.0)
        vy = float(vy or 0.0)
        displacement_magnitude = drift_intensity * deformation_factor * 0.15

        x = node["base_x"] + vx * displacement_magnitude
        y = node["base_y"] + vy * displacement_magnitude

        deformed.append({
            **node,
            "x": clamp(x, 0.0, 1.0),
            "y": clamp(y, 0.0, 1.0),
            "deformation": displacement_magnitude,
        })
    return deformed


def _compute_edge_strength(nodes: list[dict[str, Any]], drift_intensity: float, stability: float) -> list[dict[str, Any]]:
    """Compute edges between adjacent nodes with strength based on system state.

    Edges become weaker (more transparent) under high drift, stronger under stability.

    Args:
        nodes: List of deformed nodes
        drift_intensity: System drift (0-1)
        stability: System stability (0-1)

    Returns:
        List of edges between nodes
    """
    edges = []
    num_nodes = len(nodes)

    for i in range(num_nodes):
        # Connect each node to its neighbors (circular topology)
        for j in range(i + 1, min(i + 3, num_nodes)):  # Connect to next 2 neighbors
            node_a = nodes[i]
            node_b = nodes[j]

            # Distance between nodes
            dx = node_b["x"] - node_a["x"]
            dy = node_b["y"] - node_a["y"]
            distance = math.sqrt(dx**2 + dy**2)

            # Edge strength inversely proportional to drift
            strength = clamp(stability * 0.9, 0.1, 1.0)

            edges.append({
                "id": f"edge_{i}_{j}",
                "from": node_a["id"],
                "to": node_b["id"],
                "x1": node_a["x"],
                "y1": node_a["y"],
                "x2": node_b["x"],
                "y2": node_b["y"],
                "distance": distance,
                "strength": strength,
                "opacity": clamp(0.3 + strength * 0.5, 0.1, 0.8),
                "color": _edge_color_by_drift(drift_intensity),
            })

    return edges


def _edge_color_by_drift(drift_intensity: float) -> str:
    """Select edge color based on drift intensity.

    - Low drift (stable): Cool blue
    - Medium drift (transition): Purple
    - High drift (reorganization): Warm orange

    Args:
        drift_intensity: Drift value (0-1)

    Returns:
        Color hex string
    """
    if drift_intensity < 0.35:
        # Stable: cool blue
        return "#60A5FA"
    elif drift_intensity < 0.65:
        # Transition: purple
        return "#A78BFA"
    else:
        # Reorganization: warm orange
        return "#FB923C"


def render_system_geometry_viz(
    state: Any,  # SystemState
    gate_decision: dict[str, Any] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Render structural geometry visualization of system relationships.

    This replaces the previous line chart with a geometric view that shows:
    - System structure as a network of sensors (nodes) and relationships (edges)
    - Deformation reflects stability (tight/organized) vs. drift (spread/deformed)
    - Trails show prior stable states as subtle geometric ghosts
    - Current state highlighted with emphasis

    Args:
        state: SystemState object with position, velocity, drift info
        gate_decision: Gate decision dict with verdict info
        records: List of historical records for context

    Returns:
        Dict containing geometry visualization model
    """
    # Generate base sensor nodes
    num_sensors = 8
    nodes = _generate_sensor_nodes(num_sensors)

    # Extract system metrics from state
    drift_intensity = clamp(state.drift_intensity if state else 0.0, 0.0, 1.0)
    stability = 1.0 - drift_intensity  # Inverse of drift
    velocity = state.velocity if state and hasattr(state, 'velocity') else [0.0, 0.0]

    # Apply deformation based on drift and velocity
    deformed_nodes = _apply_drift_deformation(nodes, drift_intensity, velocity)

    # Compute edges with strength based on stability
    edges = _compute_edge_strength(deformed_nodes, drift_intensity, stability)

    # Generate prior stable states as subtle trails
    prior_trails = []
    if records and len(records) > 1:
        # Show trails from 2 and 4 steps back
        for steps_back in [2, 4]:
            if len(records) > steps_back:
                prior_drift = float(records[-steps_back - 1].get("structural_drift_score", 0.0))
                prior_stability = 1.0 - prior_drift
                opacity = clamp(0.08 + (5 - steps_back) * 0.04, 0.04, 0.16)
                prior_trails.append({
                    "steps_back": steps_back,
                    "drift": prior_drift,
                    "opacity": opacity,
                    "color": f"rgba(148, 163, 184, {opacity:.3f})",
                })

    # Determine phase for visual encoding
    decision = (gate_decision or {}).get("decision", "SUPPRESS").upper() if gate_decision else "SUPPRESS"
    transition_type = "STABLE"
    if records and len(records) > 0:
        transition_type = str(records[-1].get("transition_type", "STABLE")).upper()

    phase_visual = {
        "STABLE": {
            "tone": "coherent",
            "glow": 0.12,
            "color_accent": "#60A5FA",
        },
        "TRANSITION": {
            "tone": "mobilizing",
            "glow": 0.34,
            "color_accent": "#A78BFA",
        },
        "REORGANIZATION": {
            "tone": "transformative",
            "glow": 0.52,
            "color_accent": "#FB923C",
        },
    }.get(transition_type, {
        "tone": "coherent",
        "glow": 0.12,
        "color_accent": "#60A5FA",
    })

    return {
        "component": "system_geometry_visualization",
        "type": "structural_geometry",
        "title": "System Geometry",
        "subtitle": "Structure deforms with drift; tight structure indicates stability.",
        "explanation": (
            "Nodes represent sensors/functional units. Edges represent relationships. "
            "Stable systems maintain tight, organized structure. Drifting systems show deformation. "
            "Reorganization appears as structural settling into new configuration."
        ),
        "canvas": {
            "width": 900,
            "height": 400,
            "background": "#040713",
            "radial_gradient": ["#11183A", "#0A1230", "#05070F"],
        },
        "nodes": deformed_nodes,
        "edges": edges,
        "prior_trails": prior_trails,
        "current_position": {
            "x": state.position.x if state else 0.5,
            "y": state.position.y if state else 0.5,
            "highlight": True,
        },
        "metrics": {
            "drift_intensity": drift_intensity,
            "stability": stability,
            "structure_coherence": stability,  # Tightness of structure
            "deformation_magnitude": drift_intensity,
        },
        "phase_visual": phase_visual,
        "gate_decision": decision,
        "admitted": decision == "ADMIT",
        "visual_hierarchy": {
            "nodes": "primary_focus",
            "edges": "supporting_context",
            "trails": "historical_context",
        },
    }
