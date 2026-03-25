from __future__ import annotations

from typing import Mapping


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe(mapping: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def derive_path_prototypes(
    *,
    directional_evolution: Mapping[str, object] | None,
    trajectory_shape: Mapping[str, object] | None,
    geometry: Mapping[str, object] | None = None,
    state_graph: Mapping[str, object] | None = None,
    top_k: int = 3,
) -> dict[str, object]:
    if not isinstance(directional_evolution, Mapping) or not isinstance(trajectory_shape, Mapping):
        return {"available": False, "reason": "missing_signatures", "candidate_paths": []}

    shape_scores = trajectory_shape.get("shape_scores") if isinstance(trajectory_shape.get("shape_scores"), Mapping) else {}
    directional_div = _clamp01(_safe(directional_evolution, "directional_divergence_score", 0.0))
    cosine = max(-1.0, min(1.0, _safe(directional_evolution, "change_pattern_cosine_direction", 1.0)))
    polarity = max(-1.0, min(1.0, _safe(directional_evolution, "directional_polarity_score", 0.0)))
    groups = directional_evolution.get("dominant_feature_groups") if isinstance(directional_evolution.get("dominant_feature_groups"), list) else []
    group_concentration = 0.0
    if groups:
        try:
            group_concentration = _clamp01(float(groups[0].get("relative_drift_contribution", 0.0)))
        except (AttributeError, TypeError, ValueError):
            group_concentration = 0.0
    geom_curvature = _clamp01(_safe(geometry, "curvature", 0.0))
    geom_directionality = _clamp01(_safe(geometry, "directional_consistency", 0.0))
    graph_divergence = _clamp01(_safe(state_graph, "graph_divergence_score", 0.0))
    graph_commitment = _clamp01(_safe(state_graph, "path_commitment_score", 0.0))

    prototypes = {
        "MONOTONIC_DRIFT": _clamp01(
            0.5 * _safe(shape_scores, "monotonic_collapse")
            + 0.35 * _safe(shape_scores, "steady_deterioration")
            + 0.1 * max(0.0, cosine)
            + 0.05 * max(0.0, polarity)
            + 0.1 * geom_directionality
            + 0.1 * graph_commitment
        ),
        "OSCILLATORY_INSTABILITY": _clamp01(
            0.6 * _safe(shape_scores, "oscillatory_deterioration")
            + 0.2 * directional_div
            + 0.2 * _safe(shape_scores, "divergent_failure_modes")
            + 0.1 * geom_curvature
        ),
        "STEP_SHIFT": _clamp01(
            0.6 * _safe(shape_scores, "punctuated_shift")
            + 0.3 * (1.0 - max(0.0, cosine))
            + 0.1 * max(0.0, -polarity)
            + 0.1 * geom_curvature
        ),
        "LOCALIZED_DIVERGENCE": _clamp01(
            0.4 * _safe(shape_scores, "divergent_failure_modes")
            + 0.35 * group_concentration
            + 0.25 * graph_divergence
        ),
        "SYSTEMIC_COLLAPSE": _clamp01(
            0.5 * _safe(shape_scores, "steady_deterioration")
            + 0.25 * _safe(shape_scores, "monotonic_collapse")
            + 0.2 * (1.0 - group_concentration)
            + 0.05 * max(0.0, polarity)
            + 0.15 * graph_commitment
        ),
    }

    total = sum(prototypes.values()) + 1e-12
    normalized = {k: v / total for k, v in prototypes.items()}
    ranked = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)
    selected = ranked[: max(1, top_k)]

    return {
        "available": True,
        "prototype_scores": {k: round(float(v), 6) for k, v in normalized.items()},
        "prototype_count": len([v for v in normalized.values() if v >= 0.15]),
        "candidate_paths": [
            {
                "path_label": label,
                "support": round(float(score), 6),
                "direction_signature": {
                    "cosine_direction": round(float(cosine), 6),
                    "directional_divergence": round(float(directional_div), 6),
                },
                "dominant_features": directional_evolution.get("dominant_features", [])[:3],
                "confidence": round(float(_clamp01(score * (0.65 + 0.35 * (1.0 - directional_div)))), 6),
            }
            for label, score in selected
        ],
    }
