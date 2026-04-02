from __future__ import annotations

from typing import Iterable, Mapping


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tail(values: Iterable[float] | None, size: int) -> list[float]:
    if values is None:
        return []
    seq = [float(v) for v in values]
    if not seq:
        return []
    return seq[-min(size, len(seq)) :]


def _trend(values: Iterable[float] | None, span: int = 5) -> float:
    if values is None:
        return 0.0
    seq = [float(v) for v in values]
    if len(seq) < 2:
        return 0.0
    tail = seq[-min(span, len(seq)) :]
    diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def _signal_strength(value: float, low: float, high: float) -> float:
    if high <= low:
        return _clamp01(value)
    return _clamp01((value - low) / (high - low))


def _path_score(trajectory_analysis: Mapping[str, object] | None, key: str, default: float) -> float:
    if not isinstance(trajectory_analysis, Mapping):
        return float(default)
    raw_scores = trajectory_analysis.get("path_scores")
    if not isinstance(raw_scores, Mapping):
        return float(default)
    try:
        value = float(raw_scores.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    return _clamp01(value)


def estimate_risk_horizon(
    *,
    transition_pressure_history: Iterable[float] | None,
    shock_activity_history: Iterable[float] | None,
    structural_drift_history: Iterable[float] | None,
    trajectory_analysis: Mapping[str, object] | None,
    branching_analysis: Mapping[str, object] | None,
    constraint_analysis: Mapping[str, object] | None,
    temporal_quality: Mapping[str, object] | None = None,
    temporal_features: Mapping[str, object] | None = None,
    geometry: Mapping[str, object] | None = None,
    state_space_statistics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pressure_tail = _tail(transition_pressure_history, 6)
    shock_tail = _tail(shock_activity_history, 6)
    drift_tail = _tail(structural_drift_history, 6)

    has_inputs = any((pressure_tail, shock_tail, drift_tail, trajectory_analysis, branching_analysis, constraint_analysis))
    if not has_inputs:
        return {
            "available": False,
            "reason": "insufficient_inputs",
            "risk_horizon": "MID_TERM",
            "horizon_score": 0.5,
            "rationale": {
                "observation": "Horizon estimation unavailable because required observables were missing.",
            },
        }

    latest_pressure = _clamp01((pressure_tail[-1] if pressure_tail else 0.0) / 1.2)
    pressure_persistence = _clamp01(
        float(sum(1 for v in pressure_tail if v >= 0.85)) / float(max(1, len(pressure_tail)))
    )
    pressure_trend = _trend(pressure_tail)

    shock_recent = _clamp01(max(_tail(shock_tail, 3), default=0.0))
    shock_recurrence = _clamp01(sum(shock_tail) / max(1, len(shock_tail)))

    drift_trend = _trend(drift_tail)
    worsening_index = _clamp01(
        0.55 * _clamp01(pressure_trend / 0.12) + 0.45 * _clamp01(drift_trend / 0.12)
    )

    diverging = _path_score(trajectory_analysis, "diverging", 0.3333)
    metastable = _path_score(trajectory_analysis, "metastable", 0.3333)
    stabilizing = _path_score(trajectory_analysis, "stabilizing", 0.3333)

    commitment = "LOW"
    branch_tension = 0.5
    if isinstance(branching_analysis, Mapping):
        commitment = str(branching_analysis.get("commitment", "LOW")).upper()
        try:
            branch_tension = _clamp01(float(branching_analysis.get("branch_tension", 0.5)))
        except (TypeError, ValueError):
            branch_tension = 0.5

    lock_in_score = 0.0
    recovery_margin = 0.5
    if isinstance(constraint_analysis, Mapping):
        try:
            lock_in_score = _clamp01(float(constraint_analysis.get("lock_in_score", 0.0) or 0.0))
        except (TypeError, ValueError):
            lock_in_score = 0.0
        recovery_raw = constraint_analysis.get("recovery_margin", 0.5)
        try:
            recovery_margin = _clamp01(float(recovery_raw if recovery_raw is not None else 0.5))
        except (TypeError, ValueError):
            recovery_margin = 0.5

    commitment_boost = 1.0 if commitment == "HIGH" else (0.6 if commitment == "MODERATE" else 0.25)
    temporal_consistency = _clamp01(float((temporal_quality or {}).get("temporal_consistency_score", 1.0)))
    temporal_irregularity = _clamp01(float((temporal_quality or {}).get("timestamp_gap_irregularity", 0.0)))
    drift_acceleration = _clamp01(float((temporal_features or {}).get("drift_acceleration", 0.0)))
    local_velocity = _clamp01(float((geometry or {}).get("local_velocity_norm", 0.0)))
    local_acceleration = _clamp01(float((geometry or {}).get("local_acceleration_norm", 0.0)))
    state_contraction = _clamp01(float((state_space_statistics or {}).get("state_contraction_score", 0.0)))
    path_commitment = _clamp01(float((state_space_statistics or {}).get("path_commitment_score", 0.0)))
    directional_consistency = _clamp01(float((geometry or {}).get("directional_consistency", 1.0)))

    instability_build = _clamp01(
        0.23 * latest_pressure
        + 0.18 * pressure_persistence
        + 0.12 * shock_recurrence
        + 0.14 * diverging
        + 0.10 * drift_acceleration
        + 0.10 * state_contraction
        + 0.08 * branch_tension
        + 0.05 * (1.0 - directional_consistency)
    )
    directional_consensus = _clamp01(
        0.45 * diverging + 0.25 * pressure_persistence + 0.20 * lock_in_score + 0.10 * branch_tension
    )
    recovery_cap = _clamp01(
        0.45 * recovery_margin
        + 0.25 * stabilizing
        + 0.15 * temporal_consistency
        + 0.10 * max(0.0, 1.0 - state_contraction)
        + 0.05 * max(0.0, directional_consistency)
    )

    imminence_score = _clamp01(
        0.2 * shock_recent
        + 0.2 * diverging
        + 0.15 * commitment_boost
        + 0.15 * lock_in_score
        + 0.1 * latest_pressure
        + 0.1 * pressure_persistence
        + 0.1 * worsening_index
        + 0.08 * drift_acceleration
        + 0.05 * temporal_irregularity
        - 0.05 * temporal_consistency
        + 0.06 * local_velocity
        + 0.08 * local_acceleration
        + 0.08 * state_contraction
    )

    mixed_branching = _clamp01(
        0.5 * metastable + 0.3 * branch_tension + 0.2 * (1.0 - abs(diverging - stabilizing))
    )
    recovery_buffer = _clamp01(0.6 * recovery_margin + 0.4 * stabilizing)
    near_vote_count = sum(
        1
        for condition in (
            instability_build >= 0.54,
            worsening_index >= 0.42,
            drift_acceleration >= 0.35,
            lock_in_score >= 0.34,
            pressure_persistence >= 0.50,
            state_contraction >= 0.48,
            directional_consensus >= 0.50,
            recovery_cap <= 0.46,
        )
        if condition
    )
    imminent_vote_count = sum(
        1
        for condition in (
            shock_recent >= 0.80,
            instability_build >= 0.70,
            directional_consensus >= 0.66,
            lock_in_score >= 0.60,
            state_contraction >= 0.60,
            recovery_cap <= 0.34,
            branch_tension >= 0.62,
        )
        if condition
    )

    near_strength = _clamp01(
        0.35 * _signal_strength(instability_build, 0.46, 0.78)
        + 0.20 * _signal_strength(worsening_index, 0.35, 0.72)
        + 0.15 * _signal_strength(drift_acceleration, 0.28, 0.62)
        + 0.15 * _signal_strength(lock_in_score, 0.28, 0.64)
        + 0.15 * _signal_strength(1.0 - recovery_cap, 0.28, 0.64)
    )
    mid_strength = _clamp01(
        0.42 * _signal_strength(instability_build, 0.32, 0.62)
        + 0.22 * _signal_strength(mixed_branching, 0.42, 0.70)
        + 0.18 * _signal_strength(branch_tension, 0.36, 0.66)
        + 0.18 * _signal_strength(worsening_index, 0.22, 0.54)
    )
    longer_strength = _clamp01(
        0.42 * _signal_strength(recovery_cap, 0.50, 0.84)
        + 0.26 * _signal_strength(stabilizing, 0.42, 0.78)
        + 0.18 * _signal_strength(1.0 - instability_build, 0.38, 0.70)
        + 0.14 * _signal_strength(1.0 - pressure_persistence, 0.32, 0.70)
    )

    if (
        (imminent_vote_count >= 4 and near_vote_count >= 5 and imminence_score >= 0.68)
        or (imminence_score >= 0.82 and diverging >= 0.70 and lock_in_score >= 0.70 and pressure_persistence >= 0.75)
    ):
        risk_horizon = "IMMINENT"
    elif near_vote_count >= 5 and near_strength >= 0.56:
        risk_horizon = "NEAR_TERM"
    elif longer_strength >= 0.58 and near_vote_count <= 3 and mixed_branching < 0.62:
        risk_horizon = "LONGER_HORIZON"
    elif mid_strength >= 0.45 or mixed_branching >= 0.52:
        risk_horizon = "MID_TERM"
    elif imminence_score >= 0.78:
        risk_horizon = "IMMINENT"
    elif imminence_score >= 0.56:
        risk_horizon = "NEAR_TERM"
    elif imminence_score >= 0.33:
        risk_horizon = "MID_TERM"
    else:
        risk_horizon = "LONGER_HORIZON"

    return {
        "available": True,
        "risk_horizon": risk_horizon,
        "horizon_score": round(float(imminence_score), 4),
        "rationale": {
            "transition_pressure": round(float(latest_pressure), 4),
            "transition_persistence": round(float(pressure_persistence), 4),
            "shock_recency": round(float(shock_recent), 4),
            "shock_recurrence": round(float(shock_recurrence), 4),
            "worsening_index": round(float(worsening_index), 4),
            "diverging_dominance": round(float(diverging), 4),
            "metastable_dominance": round(float(metastable), 4),
            "stabilizing_dominance": round(float(stabilizing), 4),
            "branch_commitment": commitment,
            "branch_tension": round(float(branch_tension), 4),
            "lock_in_score": round(float(lock_in_score), 4),
            "recovery_margin": round(float(recovery_margin), 4),
            "recovery_buffer": round(float(recovery_buffer), 4),
            "mixed_branching_index": round(float(mixed_branching), 4),
            "temporal_consistency": round(float(temporal_consistency), 4),
            "temporal_irregularity": round(float(temporal_irregularity), 4),
            "drift_acceleration": round(float(drift_acceleration), 4),
            "local_velocity_norm": round(float(local_velocity), 4),
            "local_acceleration_norm": round(float(local_acceleration), 4),
            "state_contraction_score": round(float(state_contraction), 4),
            "path_commitment": round(float(path_commitment), 4),
            "directional_consistency": round(float(directional_consistency), 4),
            "instability_buildup": round(float(instability_build), 4),
            "directional_consensus": round(float(directional_consensus), 4),
            "recovery_cap": round(float(recovery_cap), 4),
            "near_vote_count": int(near_vote_count),
            "imminent_vote_count": int(imminent_vote_count),
            "near_strength": round(float(near_strength), 4),
            "mid_strength": round(float(mid_strength), 4),
            "longer_strength": round(float(longer_strength), 4),
        },
    }
