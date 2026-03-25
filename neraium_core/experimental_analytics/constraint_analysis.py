from __future__ import annotations

from typing import Iterable


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _last(values: Iterable[float] | None, default: float = 0.0) -> float:
    if values is None:
        return float(default)
    seq = [float(v) for v in values]
    if not seq:
        return float(default)
    return float(seq[-1])


def _tail(values: Iterable[float] | None, size: int) -> list[float]:
    if values is None:
        return []
    seq = [float(v) for v in values]
    if not seq:
        return []
    return seq[-min(size, len(seq)) :]


def _fraction_above(values: Iterable[float], threshold: float) -> float:
    seq = [float(v) for v in values]
    if not seq:
        return 0.0
    return float(sum(1 for v in seq if v >= threshold)) / float(len(seq))


def analyze_constraint_lock_in(
    *,
    transition_pressure_history: Iterable[float] | None,
    shock_activity_history: Iterable[float] | None,
    structural_drift_score: float | None,
    regime_novelty: float | None,
    regime_distance: float | None,
    subsystem_instability: float | None,
    reversibility_classification: str | None,
    reversibility_score: float | None,
    trajectory_analysis: dict[str, object] | None,
    temporal_quality: dict[str, object] | None = None,
    temporal_features: dict[str, object] | None = None,
) -> dict[str, object]:
    pressure_tail = _tail(transition_pressure_history, 6)
    shock_tail = _tail(shock_activity_history, 6)

    has_primary_inputs = any(
        (
            bool(pressure_tail),
            structural_drift_score is not None,
            subsystem_instability is not None,
            regime_novelty is not None,
            bool(shock_tail),
        )
    )
    if not has_primary_inputs:
        return {
            "available": False,
            "reason": "insufficient_inputs",
            "recovery_margin": None,
            "lock_in_score": None,
            "commitment_to_failure": "MODERATE",
            "point_of_no_return_risk": "ELEVATED",
            "rationale": {
                "observation": "Constraint analysis unavailable because required observables were missing.",
            },
        }

    pressure = _clamp01(_last(pressure_tail, 0.0) / 1.2)
    pressure_persistence = _fraction_above(pressure_tail, 0.85)

    shock_recent = _clamp01(max(_tail(shock_tail, 3), default=0.0))
    shock_recurrence = _clamp01(sum(shock_tail) / max(1, len(shock_tail)))

    drift = _clamp01(float(structural_drift_score or 0.0) / 1.2)
    novelty = _clamp01(float(regime_novelty or 0.0))
    distance = _clamp01(float(regime_distance or 0.0) / 1.5)
    regime_displacement = max(novelty, distance)
    fragmentation = _clamp01(float(subsystem_instability or 0.0) / 1.5)

    rev_class = str(reversibility_classification or "").upper()
    locked_in_index = _clamp01(float(reversibility_score or 0.0))
    if rev_class == "LOCKED_IN":
        locked_in_index = max(locked_in_index, 0.75)
    elif rev_class == "METASTABLE":
        locked_in_index = max(locked_in_index, 0.4)

    trajectory_scores = {}
    if isinstance(trajectory_analysis, dict):
        maybe_scores = trajectory_analysis.get("path_scores")
        if isinstance(maybe_scores, dict):
            trajectory_scores = maybe_scores

    stabilizing_dominance = _clamp01(float(trajectory_scores.get("stabilizing", 0.3333)))
    metastable_dominance = _clamp01(float(trajectory_scores.get("metastable", 0.3333)))
    diverging_dominance = _clamp01(float(trajectory_scores.get("diverging", 0.3333)))

    reversibility_capacity = _clamp01(1.0 - locked_in_index)
    temporal_consistency = _clamp01(float((temporal_quality or {}).get("temporal_consistency_score", 1.0)))
    temporal_irregularity = _clamp01(float((temporal_quality or {}).get("timestamp_gap_irregularity", 0.0)))
    drift_acceleration = _clamp01(float((temporal_features or {}).get("drift_acceleration", 0.0)))
    timing_instability = _clamp01(float((temporal_features or {}).get("timing_instability", temporal_irregularity)))

    recovery_margin = _clamp01(
        0.22 * (1.0 - pressure)
        + 0.16 * (1.0 - pressure_persistence)
        + 0.16 * (1.0 - drift)
        + 0.13 * (1.0 - shock_recurrence)
        + 0.08 * (1.0 - shock_recent)
        + 0.09 * (1.0 - fragmentation)
        + 0.06 * (1.0 - regime_displacement)
        + 0.06 * stabilizing_dominance
        + 0.04 * reversibility_capacity
        + 0.04 * temporal_consistency
    )

    lock_in_score = _clamp01(
        0.16 * pressure
        + 0.18 * pressure_persistence
        + 0.15 * drift
        + 0.10 * shock_recent
        + 0.10 * shock_recurrence
        + 0.10 * regime_displacement
        + 0.11 * fragmentation
        + 0.06 * diverging_dominance
        + 0.04 * metastable_dominance
        + 0.05 * timing_instability
        + 0.05 * drift_acceleration
    )

    if lock_in_score >= 0.7 or (lock_in_score >= 0.62 and pressure_persistence >= 0.65):
        commitment = "HIGH"
    elif lock_in_score >= 0.4:
        commitment = "MODERATE"
    else:
        commitment = "LOW"

    point_of_no_return_index = _clamp01(
        0.55 * lock_in_score
        + 0.35 * (1.0 - recovery_margin)
        + 0.10 * max(0.0, lock_in_score - recovery_margin)
    )
    if point_of_no_return_index >= 0.72:
        point_of_no_return_risk = "HIGH"
    elif point_of_no_return_index >= 0.42:
        point_of_no_return_risk = "ELEVATED"
    else:
        point_of_no_return_risk = "LOW"

    return {
        "available": True,
        "recovery_margin": round(float(recovery_margin), 4),
        "lock_in_score": round(float(lock_in_score), 4),
        "commitment_to_failure": commitment,
        "point_of_no_return_risk": point_of_no_return_risk,
        "rationale": {
            "transition_pressure": round(float(pressure), 4),
            "transition_persistence": round(float(pressure_persistence), 4),
            "shock_recency": round(float(shock_recent), 4),
            "shock_recurrence": round(float(shock_recurrence), 4),
            "structural_drift": round(float(drift), 4),
            "regime_displacement": round(float(regime_displacement), 4),
            "fragmentation": round(float(fragmentation), 4),
            "reversibility_capacity": round(float(reversibility_capacity), 4),
            "stabilizing_dominance": round(float(stabilizing_dominance), 4),
            "diverging_dominance": round(float(diverging_dominance), 4),
            "temporal_consistency": round(float(temporal_consistency), 4),
            "temporal_irregularity": round(float(temporal_irregularity), 4),
            "timing_instability": round(float(timing_instability), 4),
            "drift_acceleration": round(float(drift_acceleration), 4),
        },
    }
