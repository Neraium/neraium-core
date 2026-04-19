from __future__ import annotations

from typing import Iterable, Mapping


_PATH_ORDER = ("STABILIZING", "METASTABLE", "DIVERGING")
_COMMITMENT_ORDER = ("LOW", "MODERATE", "HIGH")
_LOCK_IN_ORDER = ("LOW", "ELEVATED", "HIGH")
_HORIZON_ORDER = ("LONGER_HORIZON", "MID_TERM", "NEAR_TERM", "IMMINENT")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tail(values: Iterable[float] | None, size: int) -> list[float]:
    if values is None:
        return []
    seq = [float(v) for v in values]
    if not seq:
        return []
    return seq[-min(size, len(seq)) :]


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def _safe_float(mapping: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(mapping, Mapping):
        return float(default)
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _class_from_index(index: float, levels: tuple[str, str, str], low: float, high: float) -> str:
    x = _clamp01(index)
    if x >= high:
        return levels[2]
    if x >= low:
        return levels[1]
    return levels[0]


def _horizon_from_imminence(imminence: float) -> str:
    x = _clamp01(imminence)
    if x >= 0.8:
        return "IMMINENT"
    if x >= 0.58:
        return "NEAR_TERM"
    if x >= 0.34:
        return "MID_TERM"
    return "LONGER_HORIZON"


def _shift_label(before: str, after: str, ordering: tuple[str, ...], improve_direction: int) -> str:
    if before not in ordering or after not in ordering:
        return "CHANGED" if before != after else "UNCHANGED"
    i0 = ordering.index(before)
    i1 = ordering.index(after)
    if i0 == i1:
        return "UNCHANGED"
    movement = (i0 - i1) * improve_direction
    if movement >= 2:
        return "IMPROVED_STRONGLY"
    if movement == 1:
        return "IMPROVED"
    if movement <= -2:
        return "WORSENED_STRONGLY"
    return "WORSENED"


def _project_state(
    *,
    pressure: float,
    pressure_trend: float,
    shock: float,
    drift: float,
    drift_trend: float,
    stabilizing_score: float,
    metastable_score: float,
    diverging_score: float,
    branch_tension: float,
    top_two_margin: float,
    lock_in_score: float,
    recovery_margin: float,
    commitment_seed: float,
    horizon_hint: str | None,
    dominant_branch: str | None = None,
    alternative_branch: str | None = None,
) -> dict[str, str]:
    worsening_index = _clamp01(0.55 * _clamp01(pressure_trend / 0.12) + 0.45 * _clamp01(drift_trend / 0.12))
    commitment_index = _clamp01(
        0.26 * diverging_score
        + 0.24 * lock_in_score
        + 0.2 * pressure
        + 0.15 * (1.0 - recovery_margin)
        + 0.15 * commitment_seed
    )
    lock_risk_index = _clamp01(
        0.35 * lock_in_score
        + 0.22 * pressure
        + 0.18 * diverging_score
        + 0.15 * (1.0 - recovery_margin)
        + 0.1 * shock
    )
    imminence = _clamp01(
        0.22 * shock
        + 0.2 * diverging_score
        + 0.18 * commitment_index
        + 0.15 * lock_risk_index
        + 0.15 * pressure
        + 0.1 * worsening_index
    )

    if diverging_score >= 0.52 and commitment_index >= 0.58 and lock_risk_index >= 0.58:
        projected_path = "DIVERGING"
    elif branch_tension >= 0.62 and top_two_margin <= 0.12:
        projected_path = "METASTABLE"
    elif recovery_margin >= 0.62 and lock_risk_index <= 0.45 and stabilizing_score >= diverging_score:
        projected_path = "STABILIZING"
    else:
        projected_path = max(
            (
                ("STABILIZING", stabilizing_score),
                ("METASTABLE", metastable_score),
                ("DIVERGING", diverging_score),
            ),
            key=lambda item: item[1],
        )[0]

    projected_commitment = _class_from_index(commitment_index, _COMMITMENT_ORDER, low=0.42, high=0.72)
    projected_lock_in_risk = _class_from_index(lock_risk_index, _LOCK_IN_ORDER, low=0.4, high=0.72)
    projected_horizon = _horizon_from_imminence(imminence)

    if str(horizon_hint or "").upper() == "IMMINENT" and imminence >= 0.72:
        projected_horizon = "IMMINENT"

    return {
        "projected_path": projected_path,
        "projected_branch": dominant_branch or projected_path,
        "alternative_projected_branch": alternative_branch or projected_path,
        "projected_commitment": projected_commitment,
        "projected_lock_in_risk": projected_lock_in_risk,
        "projected_horizon": projected_horizon,
    }


def simulate_counterfactual_futures(
    transition_pressure_history: Iterable[float] | None,
    shock_activity_history: Iterable[float] | None,
    structural_drift_history: Iterable[float] | None,
    trajectory_analysis: Mapping[str, object] | None = None,
    branching_analysis: Mapping[str, object] | None = None,
    constraint_analysis: Mapping[str, object] | None = None,
    hierarchy_analysis: Mapping[str, object] | None = None,
    horizon_analysis: Mapping[str, object] | None = None,
    temporal_quality: Mapping[str, object] | None = None,
    temporal_features: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pressure_tail = _tail(transition_pressure_history, 6)
    shock_tail = _tail(shock_activity_history, 6)
    drift_tail = _tail(structural_drift_history, 6)
    if not pressure_tail or not shock_tail or not drift_tail:
        return {
            "available": False,
            "reason": "insufficient_required_histories",
            "baseline_future": None,
            "counterfactuals": [],
            "rationale": {
                "observation": "Counterfactual simulation unavailable because required histories were missing.",
                "dominant_drivers": [],
                "sensitivity_summary": "No deterministic projection can be produced without transition, shock, and drift histories.",
            },
        }

    pressure = _clamp01(pressure_tail[-1] / 1.2)
    pressure_trend = _trend(pressure_tail)
    shock_recent = _clamp01(max(shock_tail[-3:], default=0.0))
    shock_activity = _clamp01(sum(shock_tail) / max(1, len(shock_tail)))
    shock = _clamp01(0.6 * shock_recent + 0.4 * shock_activity)
    drift = _clamp01(drift_tail[-1] / 1.2)
    drift_trend = _trend(drift_tail)

    path_scores = trajectory_analysis.get("path_scores") if isinstance(trajectory_analysis, Mapping) else {}
    stabilizing = _clamp01(_safe_float(path_scores, "stabilizing", 0.3333))
    metastable = _clamp01(_safe_float(path_scores, "metastable", 0.3333))
    diverging = _clamp01(_safe_float(path_scores, "diverging", 0.3333))

    branch_tension = _clamp01(_safe_float(branching_analysis, "branch_tension", 0.5))
    top_two_margin = _clamp01(_safe_float(branching_analysis, "top_two_margin", 0.35))
    dominant_branch = str((branching_analysis or {}).get("dominant_branch", "")).upper() or None
    alternative_branch = str((branching_analysis or {}).get("alternative_branch", "")).upper() or None

    lock_in_score = _clamp01(_safe_float(constraint_analysis, "lock_in_score", 0.0))
    recovery_margin = _clamp01(_safe_float(constraint_analysis, "recovery_margin", 0.5))

    commitment_text = str((branching_analysis or {}).get("commitment", "LOW")).upper()
    commitment_seed = 1.0 if commitment_text == "HIGH" else (0.6 if commitment_text == "MODERATE" else 0.25)

    horizon_hint = str((horizon_analysis or {}).get("risk_horizon", "")).upper() if isinstance(horizon_analysis, Mapping) else ""
    temporal_consistency = _clamp01(_safe_float(temporal_quality, "temporal_consistency_score", 1.0))
    temporal_irregularity = _clamp01(_safe_float(temporal_quality, "timestamp_gap_irregularity", 0.0))
    drift_acceleration = _clamp01(_safe_float(temporal_features, "drift_acceleration", 0.0))
    timing_intensity = _clamp01(_safe_float(temporal_features, "timing_intensity", 0.0))

    if isinstance(hierarchy_analysis, Mapping):
        propagation_risk = str(hierarchy_analysis.get("propagation_risk", "LOW")).upper()
        if propagation_risk == "HIGH":
            diverging = _clamp01(diverging + 0.05)
            stabilizing = _clamp01(stabilizing - 0.03)
        elif propagation_risk == "LOW":
            stabilizing = _clamp01(stabilizing + 0.02)

    total = max(1e-9, stabilizing + metastable + diverging)
    stabilizing, metastable, diverging = stabilizing / total, metastable / total, diverging / total

    baseline = _project_state(
        pressure=pressure,
        pressure_trend=pressure_trend + 0.06 * drift_acceleration,
        shock=shock,
        drift=drift,
        drift_trend=drift_trend + 0.08 * timing_intensity,
        stabilizing_score=stabilizing,
        metastable_score=metastable,
        diverging_score=diverging,
        branch_tension=branch_tension,
        top_two_margin=top_two_margin,
        lock_in_score=lock_in_score,
        recovery_margin=recovery_margin,
        commitment_seed=commitment_seed,
        horizon_hint=horizon_hint,
        dominant_branch=dominant_branch,
        alternative_branch=alternative_branch,
    )

    scenarios = [
        {
            "scenario": "pressure_relief",
            "assumption": "transition pressure reduced while shock activity remains bounded",
            "pressure": _clamp01(pressure * 0.76),
            "pressure_trend": pressure_trend - 0.06,
            "shock": _clamp01(shock * 0.8),
            "drift": _clamp01(drift * 0.88),
            "drift_trend": drift_trend - 0.05,
            "stabilizing": _clamp01(stabilizing + 0.08),
            "metastable": _clamp01(metastable + 0.02),
            "diverging": _clamp01(diverging - 0.1),
            "branch_tension": _clamp01(branch_tension - 0.07),
            "top_two_margin": _clamp01(top_two_margin + 0.03),
            "lock_in_score": _clamp01(lock_in_score * 0.82),
            "recovery_margin": _clamp01(recovery_margin + 0.12),
            "commitment_seed": max(0.0, commitment_seed - 0.2),
        },
        {
            "scenario": "stabilized_structure",
            "assumption": "drift and branching tension soften and recovery margin improves",
            "pressure": _clamp01(pressure * 0.9),
            "pressure_trend": pressure_trend - 0.04,
            "shock": _clamp01(shock * 0.9),
            "drift": _clamp01(drift * 0.78),
            "drift_trend": drift_trend - 0.08,
            "stabilizing": _clamp01(stabilizing + 0.12),
            "metastable": _clamp01(metastable - 0.05),
            "diverging": _clamp01(diverging - 0.1),
            "branch_tension": _clamp01(branch_tension - 0.14),
            "top_two_margin": _clamp01(top_two_margin + 0.06),
            "lock_in_score": _clamp01(lock_in_score * 0.75),
            "recovery_margin": _clamp01(recovery_margin + 0.18),
            "commitment_seed": max(0.0, commitment_seed - 0.28),
        },
        {
            "scenario": "continued_degradation",
            "assumption": "transition pressure persists or worsens and branching resolves toward divergence",
            "pressure": _clamp01(min(1.0, pressure * 1.14 + 0.04)),
            "pressure_trend": pressure_trend + 0.05,
            "shock": _clamp01(min(1.0, shock * 1.15 + 0.03)),
            "drift": _clamp01(min(1.0, drift * 1.12 + 0.04)),
            "drift_trend": drift_trend + 0.06,
            "stabilizing": _clamp01(stabilizing - 0.09),
            "metastable": _clamp01(metastable - 0.02),
            "diverging": _clamp01(diverging + 0.14),
            "branch_tension": _clamp01(max(branch_tension - 0.08, 0.0)),
            "top_two_margin": _clamp01(top_two_margin + 0.06),
            "lock_in_score": _clamp01(min(1.0, lock_in_score * 1.18 + 0.04)),
            "recovery_margin": _clamp01(max(0.0, recovery_margin - 0.2)),
            "commitment_seed": min(1.0, commitment_seed + 0.26),
        },
    ]

    counterfactuals: list[dict[str, object]] = []
    for scenario in scenarios:
        projected = _project_state(
            pressure=float(scenario["pressure"]),
            pressure_trend=float(scenario["pressure_trend"]) + 0.04 * drift_acceleration,
            shock=float(scenario["shock"]),
            drift=float(scenario["drift"]),
            drift_trend=float(scenario["drift_trend"]) + 0.05 * timing_intensity,
            stabilizing_score=float(scenario["stabilizing"]),
            metastable_score=float(scenario["metastable"]),
            diverging_score=float(scenario["diverging"]),
            branch_tension=float(scenario["branch_tension"]),
            top_two_margin=float(scenario["top_two_margin"]),
            lock_in_score=float(scenario["lock_in_score"]),
            recovery_margin=float(scenario["recovery_margin"]),
            commitment_seed=float(scenario["commitment_seed"]),
            horizon_hint=horizon_hint,
            dominant_branch=dominant_branch,
            alternative_branch=alternative_branch,
        )
        if horizon_hint == "IMMINENT" and scenario["scenario"] == "continued_degradation":
            projected["projected_horizon"] = "IMMINENT"
        counterfactuals.append(
            {
                "scenario": scenario["scenario"],
                "assumption": scenario["assumption"],
                **projected,
                "delta_vs_baseline": {
                    "branch_shift": _shift_label(
                        baseline["projected_branch"],
                        projected["projected_branch"],
                        _PATH_ORDER,
                        improve_direction=1,
                    ),
                    "path_shift": _shift_label(
                        baseline["projected_path"],
                        projected["projected_path"],
                        _PATH_ORDER,
                        improve_direction=1,
                    ),
                    "commitment_shift": _shift_label(
                        baseline["projected_commitment"],
                        projected["projected_commitment"],
                        _COMMITMENT_ORDER,
                        improve_direction=1,
                    ),
                    "lock_in_shift": _shift_label(
                        baseline["projected_lock_in_risk"],
                        projected["projected_lock_in_risk"],
                        _LOCK_IN_ORDER,
                        improve_direction=1,
                    ),
                    "horizon_shift": _shift_label(
                        baseline["projected_horizon"],
                        projected["projected_horizon"],
                        _HORIZON_ORDER,
                        improve_direction=1,
                    ),
                },
            }
        )

    dominant_drivers = ["transition_pressure", "structural_drift", "branching_tension"]
    if lock_in_score >= 0.55:
        dominant_drivers.append("lock_in_risk")
    if shock >= 0.5:
        dominant_drivers.append("shock_activity")

    observation = (
        "Short-horizon projection reflects current transition pressure, drift direction, and branching lock-in. "
        "Counterfactual scenarios only perturb those observed structural drivers deterministically."
    )

    return {
        "available": True,
        "reason": None,
        "baseline_future": baseline,
        "counterfactuals": counterfactuals,
        "rationale": {
            "observation": observation,
            "dominant_drivers": dominant_drivers,
            "sensitivity_summary": (
                "Pressure/drift relief biases toward stabilizing or less committed futures; persistent degradation "
                "biases toward higher commitment, higher lock-in, and shorter horizon."
            ),
            "temporal_counterfactual_trust": round(float(_clamp01(0.7 * temporal_consistency + 0.3 * (1.0 - temporal_irregularity))), 4),
            "timing_intensity": round(float(timing_intensity), 4),
        },
    }
