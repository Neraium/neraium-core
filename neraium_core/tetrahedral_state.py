from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


_TETRA_KEYS = (
    "structural_drift_score",
    "relational_instability_score",
    "transition_pressure",
    "temporal_inconsistency",
)

_VERTEX_LABELS = {
    "structural_drift_score": "STRUCTURAL",
    "relational_instability_score": "RELATIONAL",
    "transition_pressure": "TRANSITION",
    "temporal_inconsistency": "TEMPORAL",
}

_FACE_LABELS = {
    "structural_drift_score": "RELATIONAL_TRANSITION_TEMPORAL",
    "relational_instability_score": "STRUCTURAL_TRANSITION_TEMPORAL",
    "transition_pressure": "STRUCTURAL_RELATIONAL_TEMPORAL",
    "temporal_inconsistency": "STRUCTURAL_RELATIONAL_TRANSITION",
}

_INTERPRETED_LABELS = {
    "STRUCTURAL": "STRUCTURAL_STRESS_BUILDING",
    "RELATIONAL": "COUPLING_BREAKDOWN",
    "TRANSITION": "ACTIVE_TRANSITION",
    "TEMPORAL": "TEMPORAL_DEGRADATION",
}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_tetrahedral_weights(
    structural_drift_score: float,
    relational_instability_score: float,
    transition_pressure: float,
    temporal_consistency_score: float,
) -> dict[str, float]:
    """Return deterministic normalized tetrahedral weights."""
    values = {
        "structural_drift_score": _clamp01(float(structural_drift_score)),
        "relational_instability_score": _clamp01(float(relational_instability_score)),
        "transition_pressure": _clamp01(float(transition_pressure)),
        "temporal_inconsistency": _clamp01(1.0 - float(temporal_consistency_score)),
    }
    total = float(sum(values.values()))
    if total <= 1e-12:
        return {k: 0.25 for k in _TETRA_KEYS}
    return {k: float(values[k] / total) for k in _TETRA_KEYS}


def weights_to_position(weights: Mapping[str, float]) -> list[float]:
    """Map tetrahedral weights to a 3D point using fixed tetrahedron vertices."""
    w = {k: float(weights.get(k, 0.0)) for k in _TETRA_KEYS}
    vertices = {
        "structural_drift_score": np.array([1.0, 1.0, 1.0]),
        "relational_instability_score": np.array([1.0, -1.0, -1.0]),
        "transition_pressure": np.array([-1.0, 1.0, -1.0]),
        "temporal_inconsistency": np.array([-1.0, -1.0, 1.0]),
    }
    point = sum(w[key] * vertices[key] for key in _TETRA_KEYS)
    return [float(point[0]), float(point[1]), float(point[2])]


def compute_motion_features(positions: Sequence[Sequence[float]]) -> dict[str, float | str]:
    """Derive simple speed/curvature from recent tetrahedral positions."""
    if len(positions) < 2:
        return {"speed": 0.0, "curvature": 0.0, "movement_summary": "stationary"}

    arr = np.asarray(positions, dtype=float)
    delta = arr[-1] - arr[-2]
    speed = float(np.linalg.norm(delta))

    curvature = 0.0
    if len(arr) >= 3:
        prev = arr[-2] - arr[-3]
        prev_norm = float(np.linalg.norm(prev))
        delta_norm = float(np.linalg.norm(delta))
        if prev_norm > 1e-12 and delta_norm > 1e-12:
            cos_theta = float(np.dot(prev, delta) / (prev_norm * delta_norm))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            curvature = float(np.arccos(cos_theta) / np.pi)

    # Updated terminology: explicitly mark as geometric_* to clarify these are spatial motions in tetrahedral space,
    # not state transitions or behavioral changes
    if speed < 0.02:
        movement_summary = "geometric_stationary"
    elif curvature > 0.35:
        movement_summary = "geometric_turning"
    else:
        movement_summary = "steady_drift"

    return {
        "speed": float(speed),
        "curvature": float(curvature),
        "movement_summary": movement_summary,
    }


def compute_semantic_consistency_flags(
    state_label: str,
    motion_summary: str,
    policy_state: str | None = None,
    transition_state: str | None = None,
    drift_score: float | None = None,
) -> dict[str, object]:
    """Assess semantic consistency between tetrahedral geometry and system state.

    Flags contradictions that may confuse users (e.g., ALERT state but BALANCED geometry).
    """
    consistency_status = "coherent"
    tension_type: str | None = None
    semantic_context = ""

    # Tension 1: ALERT state but geometry shows no dominance
    if (
        policy_state == "ALERT"
        and state_label == "GEOMETRICALLY_NEUTRAL"
    ):
        consistency_status = "tension"
        tension_type = "alert_but_geometrically_neutral"
        semantic_context = (
            "System is in ALERT (high instability), but structural dimensions are equally "
            "involved rather than localized to one axis. Indicates system-wide departure, not "
            "localized stress. All four metrics are significantly elevated."
        )

    # Tension 2: High drift but geometric motion is stationary
    if (
        drift_score is not None
        and drift_score >= 0.5
        and motion_summary == "geometric_stationary"
    ):
        if not tension_type:
            consistency_status = "tension"
            tension_type = "high_drift_with_geometric_stasis"
            semantic_context = (
                "System is dwelling in an elevated stress region. Drift is high, but "
                "metric values are not changing rapidly frame-to-frame, so the geometric "
                "position in tetrahedral space remains fixed. This is normal when the system "
                "is sustained at an elevated operating point."
            )

    # Tension 3: Sustained transition but geometric motion is stationary
    if (
        transition_state == "SUSTAINED_TRANSITION"
        and motion_summary == "geometric_stationary"
    ):
        if not tension_type:
            consistency_status = "tension"
            tension_type = "sustained_transition_with_geometric_stasis"
            semantic_context = (
                "System is in an active, persisting state transition. Geometric position "
                "is stable (stasis) because transition pressure remains constant. Geometric "
                "motion measures position movement in tetrahedral space, not state evolution."
            )

    return {
        "consistency_status": consistency_status,
        "tension_type": tension_type,
        "semantic_context": semantic_context,
    }


def compute_tetrahedral_state(
    *,
    structural_drift_score: float,
    relational_instability_score: float,
    transition_pressure: float,
    temporal_consistency_score: float,
    history_positions: Iterable[Sequence[float]] | None = None,
    regime_drift: float | None = None,
    reversibility: float | None = None,
    geometry_curvature: float | None = None,
    policy_state: str | None = None,
    transition_state: str | None = None,
) -> dict[str, object]:
    """Build a read-only tetrahedral state payload from existing metrics.

    Optional parameters:
    - policy_state: Current ALERT/WATCH/STABLE (used for semantic consistency checking)
    - transition_state: WARMUP/STABLE/EMERGING_TRANSITION/SUSTAINED_TRANSITION

    Includes semantic consistency flags to identify contradictions between geometry and policy.
    """
    weights = compute_tetrahedral_weights(
        structural_drift_score=structural_drift_score,
        relational_instability_score=relational_instability_score,
        transition_pressure=transition_pressure,
        temporal_consistency_score=temporal_consistency_score,
    )
    position = weights_to_position(weights)
    nearest_vertex_key = max(_TETRA_KEYS, key=lambda key: weights.get(key, 0.0))
    nearest_face_key = min(_TETRA_KEYS, key=lambda key: weights.get(key, 0.0))

    sorted_weights = sorted(weights.values(), reverse=True)
    edge_alignment = float(max(0.0, min(1.0, (sorted_weights[0] + sorted_weights[1]) - 0.5)))

    position_history = list(history_positions or []) + [position]
    motion = compute_motion_features(position_history)
    curvature = float(motion["curvature"])
    if geometry_curvature is not None:
        curvature = float(max(0.0, min(1.0, 0.65 * curvature + 0.35 * _clamp01(float(geometry_curvature)))))

    weight_peak = float(weights[nearest_vertex_key])
    # Renamed: BALANCED → GEOMETRICALLY_NEUTRAL (clearer meaning)
    state_label = "GEOMETRICALLY_NEUTRAL"
    if weight_peak >= 0.45:
        state_label = f"{_VERTEX_LABELS[nearest_vertex_key]}_DOMINANT"
    elif edge_alignment >= 0.35:
        state_label = "EDGE_ALIGNED"

    # Compute semantic consistency flags
    motion_summary_value = str(motion["movement_summary"])
    semantic_flags = compute_semantic_consistency_flags(
        state_label=state_label,
        motion_summary=motion_summary_value,
        policy_state=policy_state,
        transition_state=transition_state,
        drift_score=float(structural_drift_score),
    )

    payload = {
        "weights": {k: round(float(v), 6) for k, v in weights.items()},
        "position": [round(float(v), 6) for v in position],
        "nearest_vertex": _VERTEX_LABELS[nearest_vertex_key],
        "interpreted_label": _INTERPRETED_LABELS[_VERTEX_LABELS[nearest_vertex_key]],
        "nearest_face": _FACE_LABELS[nearest_face_key],
        "edge_alignment": round(edge_alignment, 6),
        "speed": round(float(motion["speed"]), 6),
        "curvature": round(float(curvature), 6),
        "state_label": state_label,
        "geometric_motion_class": motion_summary_value,  # New name (clearer meaning)
        "movement_summary": motion_summary_value,  # Backward compatibility
        "semantic_consistency": semantic_flags,
    }
    if regime_drift is not None:
        payload["regime_drift"] = round(_clamp01(float(regime_drift)), 6)
    if reversibility is not None:
        payload["reversibility"] = round(_clamp01(float(reversibility)), 6)
    if policy_state is not None:
        payload["policy_state_context"] = policy_state
    if transition_state is not None:
        payload["transition_state_context"] = transition_state
    return payload
