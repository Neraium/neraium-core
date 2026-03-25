from __future__ import annotations

from typing import Iterable, Mapping


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _trend(values: Iterable[float], span: int = 5) -> float:
    seq = [float(v) for v in values]
    if len(seq) < 2:
        return 0.0
    tail = seq[-min(span, len(seq)) :]
    diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def _oscillation(values: Iterable[float], span: int = 6) -> float:
    seq = [float(v) for v in values]
    if len(seq) < 4:
        return 0.0
    tail = seq[-min(span, len(seq)) :]
    diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    sign_changes = 0
    prev_sign = 0
    for d in diffs:
        sign = 1 if d > 1e-9 else (-1 if d < -1e-9 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            sign_changes += 1
        prev_sign = sign
    denom = max(1, len(diffs) - 1)
    return _clamp01(sign_changes / denom)


def _safe_float(mapping: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def classify_trajectory_path(
    *,
    drift_history: Iterable[float],
    transition_pressure_history: Iterable[float],
    subsystem_instability_history: Iterable[float],
    regime_novelty_history: Iterable[float],
    shock_activity_history: Iterable[float],
    reversibility_classification: str | None,
    reversibility_score: float | None,
    directional_evolution: Mapping[str, object] | None = None,
    trajectory_shape: Mapping[str, object] | None = None,
    path_prototypes: Mapping[str, object] | None = None,
    temporal_quality: Mapping[str, object] | None = None,
    temporal_features: Mapping[str, object] | None = None,
    geometry: Mapping[str, object] | None = None,
    state_space_statistics: Mapping[str, object] | None = None,
    state_graph: Mapping[str, object] | None = None,
) -> dict[str, object]:
    drift_values = [float(v) for v in drift_history]
    pressure_values = [float(v) for v in transition_pressure_history]
    subsystem_values = [float(v) for v in subsystem_instability_history]
    novelty_values = [float(v) for v in regime_novelty_history]
    shock_values = [float(v) for v in shock_activity_history]

    drift = _clamp01((drift_values[-1] if drift_values else 0.0) / 1.2)
    pressure = _clamp01((pressure_values[-1] if pressure_values else 0.0) / 1.2)
    subsystem = _clamp01((subsystem_values[-1] if subsystem_values else 0.0) / 1.5)
    novelty = _clamp01(novelty_values[-1] if novelty_values else 0.0)
    shock_recent = _clamp01(max(shock_values[-3:], default=0.0))
    shock_activity = _clamp01(sum(shock_values[-6:]) / max(1, len(shock_values[-6:])))
    shock = _clamp01(0.65 * shock_recent + 0.35 * shock_activity)

    drift_trend = _trend(drift_values)
    pressure_trend = _trend(pressure_values)
    subsystem_trend = _trend(subsystem_values)
    novelty_trend = _trend(novelty_values)

    rising = _clamp01(
        0.35 * _clamp01(drift_trend / 0.12)
        + 0.35 * _clamp01(pressure_trend / 0.12)
        + 0.20 * _clamp01(subsystem_trend / 0.18)
        + 0.10 * _clamp01(novelty_trend / 0.10)
    )
    falling = _clamp01(
        0.40 * _clamp01((-drift_trend) / 0.12)
        + 0.35 * _clamp01((-pressure_trend) / 0.12)
        + 0.25 * _clamp01((-subsystem_trend) / 0.18)
    )

    oscillation = _clamp01(
        0.55 * _oscillation(drift_values)
        + 0.45 * _oscillation(pressure_values)
    )
    fragility = _clamp01(0.4 * pressure + 0.35 * subsystem + 0.25 * novelty)
    ambiguous_motion = _clamp01(1.0 - abs(rising - falling))

    reversibility = (reversibility_classification or "").upper()
    locked_in = _clamp01(float(reversibility_score or 0.0))
    rev_stable = 1.0 if reversibility == "REVERSIBLE" else 0.0
    rev_meta = 1.0 if reversibility == "METASTABLE" else 0.0
    rev_locked = 1.0 if reversibility == "LOCKED_IN" else 0.0

    directional_div = _clamp01(_safe_float(directional_evolution, "directional_divergence_score", 0.0))
    cosine_dir = max(-1.0, min(1.0, _safe_float(directional_evolution, "change_pattern_cosine_direction", 1.0)))
    timing_irregularity = _clamp01(_safe_float(temporal_quality, "timestamp_gap_irregularity", 0.0))
    temporal_consistency = _clamp01(_safe_float(temporal_quality, "temporal_consistency_score", 0.0))
    timing_instability = _clamp01(_safe_float(temporal_features, "timing_instability", timing_irregularity))
    drift_velocity = _clamp01(_safe_float(temporal_features, "drift_velocity", 0.0))
    drift_acceleration = _clamp01(_safe_float(temporal_features, "drift_acceleration", 0.0))

    geom_curvature = _clamp01(_safe_float(geometry, "curvature", 0.0))
    geom_directional_consistency = _clamp01(_safe_float(geometry, "directional_consistency", 0.0))
    geom_angular_change = _clamp01(_safe_float(geometry, "angular_change", 0.0) / 3.141592653589793)
    stat_expansion = _clamp01(_safe_float(state_space_statistics, "state_expansion_score", 0.0))
    stat_contraction = _clamp01(_safe_float(state_space_statistics, "state_contraction_score", 0.0))
    graph_divergence = _clamp01(_safe_float(state_graph, "graph_divergence_score", 0.0))
    graph_commitment = _clamp01(_safe_float(state_graph, "path_commitment_score", 0.0))

    shape_scores = trajectory_shape.get("shape_scores") if isinstance(trajectory_shape, Mapping) else {}
    oscillatory_shape = _clamp01(_safe_float(shape_scores, "oscillatory_deterioration", 0.0))
    punctuated_shape = _clamp01(_safe_float(shape_scores, "punctuated_shift", 0.0))
    monotonic_shape = _clamp01(_safe_float(shape_scores, "monotonic_collapse", 0.0))
    divergent_modes = _clamp01(_safe_float(shape_scores, "divergent_failure_modes", 0.0))

    stabilizing = _clamp01(
        0.30 * (1.0 - drift)
        + 0.20 * (1.0 - pressure)
        + 0.14 * (1.0 - subsystem)
        + 0.13 * falling
        + 0.10 * max(rev_stable, 1.0 - locked_in)
        + 0.13 * max(0.0, cosine_dir)
    )
    diverging = _clamp01(
        0.20 * drift
        + 0.21 * pressure
        + 0.13 * subsystem
        + 0.13 * shock
        + 0.08 * novelty
        + 0.08 * rising
        + 0.07 * max(rev_locked, locked_in)
        + 0.10 * monotonic_shape
        + 0.06 * drift_velocity
        + 0.06 * drift_acceleration
        + 0.08 * geom_curvature
        + 0.08 * graph_divergence
        + 0.06 * stat_expansion
    )
    metastable = _clamp01(
        0.25 * fragility
        + 0.18 * oscillation
        + 0.15 * ambiguous_motion
        + 0.10 * max(rev_meta, 1.0 - abs(0.5 - locked_in) * 2.0)
        + 0.10 * (1.0 - abs(stabilizing - diverging))
        + 0.12 * oscillatory_shape
        + 0.10 * punctuated_shape
        + 0.08 * timing_irregularity
        + 0.06 * timing_instability
        + 0.06 * geom_angular_change
        + 0.05 * (1.0 - geom_directional_consistency)
    )
    stabilizing = _clamp01(
        stabilizing
        + 0.10 * temporal_consistency
        - 0.06 * timing_irregularity
        + 0.08 * geom_directional_consistency
        + 0.06 * stat_contraction
        + 0.06 * graph_commitment
    )

    total = max(1e-9, stabilizing + metastable + diverging)
    score_stabilizing = stabilizing / total
    score_metastable = metastable / total
    score_diverging = diverging / total

    if directional_div >= 0.4 or divergent_modes >= 0.35:
        score_metastable = _clamp01(score_metastable + 0.08 * directional_div + 0.07 * divergent_modes)
        score_diverging = _clamp01(score_diverging + 0.04 * max(0.0, -cosine_dir))
        norm = max(1e-9, score_stabilizing + score_metastable + score_diverging)
        score_stabilizing, score_metastable, score_diverging = (
            score_stabilizing / norm,
            score_metastable / norm,
            score_diverging / norm,
        )

    dominant = max(
        (
            ("STABILIZING", score_stabilizing),
            ("METASTABLE", score_metastable),
            ("DIVERGING", score_diverging),
        ),
        key=lambda item: item[1],
    )[0]

    candidate_paths = []
    if isinstance(path_prototypes, Mapping):
        raw_candidates = path_prototypes.get("candidate_paths")
        if isinstance(raw_candidates, list):
            candidate_paths = raw_candidates

    concentration = max(score_stabilizing, score_metastable, score_diverging)
    diversity = _clamp01(1.0 - concentration)

    rationale = {
        "drift_trend": round(float(drift_trend), 4),
        "transition_pressure_trend": round(float(pressure_trend), 4),
        "subsystem_instability_trend": round(float(subsystem_trend), 4),
        "regime_novelty_trend": round(float(novelty_trend), 4),
        "shock_recent": round(float(shock_recent), 4),
        "shock_activity": round(float(shock_activity), 4),
        "oscillation_index": round(float(oscillation), 4),
        "fragility_index": round(float(fragility), 4),
        "directional_divergence": round(float(directional_div), 4),
        "change_pattern_cosine_direction": round(float(cosine_dir), 4),
        "temporal_consistency": round(float(temporal_consistency), 4),
        "timing_irregularity": round(float(timing_irregularity), 4),
        "timing_instability": round(float(timing_instability), 4),
        "drift_velocity": round(float(drift_velocity), 4),
        "drift_acceleration": round(float(drift_acceleration), 4),
        "curvature": round(float(geom_curvature), 4),
        "directional_consistency": round(float(geom_directional_consistency), 4),
        "state_expansion": round(float(stat_expansion), 4),
        "state_contraction": round(float(stat_contraction), 4),
        "graph_divergence": round(float(graph_divergence), 4),
        "graph_commitment": round(float(graph_commitment), 4),
    }
    return {
        "dominant_path": dominant,
        "path_scores": {
            "stabilizing": round(float(score_stabilizing), 4),
            "metastable": round(float(score_metastable), 4),
            "diverging": round(float(score_diverging), 4),
        },
        "candidate_paths": candidate_paths,
        "directional_evolution": directional_evolution if isinstance(directional_evolution, Mapping) else None,
        "trajectory_shape": trajectory_shape if isinstance(trajectory_shape, Mapping) else None,
        "path_prototypes": path_prototypes if isinstance(path_prototypes, Mapping) else None,
        "diagnostics": {
            "path_diversity_score": round(float(diversity), 4),
            "dominant_path_concentration_ratio": round(float(concentration), 4),
            "directional_divergence_score": round(float(directional_div), 4),
            "shape_divergence_score": round(float(divergent_modes), 4),
            "geometric_curvature_score": round(float(geom_curvature), 4),
            "state_graph_divergence_score": round(float(graph_divergence), 4),
            "state_graph_commitment_score": round(float(graph_commitment), 4),
        },
        "rationale": rationale,
    }
