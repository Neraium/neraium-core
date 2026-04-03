from __future__ import annotations

from neraium_core.experimental_analytics.horizon_analysis import estimate_risk_horizon


def _trajectory(stabilizing: float, metastable: float, diverging: float) -> dict[str, object]:
    return {
        "dominant_path": "DIVERGING",
        "path_scores": {
            "stabilizing": stabilizing,
            "metastable": metastable,
            "diverging": diverging,
        },
    }


def test_horizon_imminent_for_strong_break_high_commitment() -> None:
    out = estimate_risk_horizon(
        transition_pressure_history=[0.7, 0.95, 1.08, 1.15, 1.24, 1.28],
        shock_activity_history=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        structural_drift_history=[0.4, 0.56, 0.69, 0.83, 0.95, 1.08],
        trajectory_analysis=_trajectory(stabilizing=0.1, metastable=0.14, diverging=0.76),
        branching_analysis={"commitment": "HIGH", "branch_tension": 0.22},
        constraint_analysis={"lock_in_score": 0.81, "recovery_margin": 0.2},
    )
    assert out["available"] is True
    assert out["risk_horizon"] == "IMMINENT"
    assert out["horizon_score"] >= 0.7


def test_horizon_near_term_for_persistent_worsening() -> None:
    out = estimate_risk_horizon(
        transition_pressure_history=[0.55, 0.74, 0.88, 0.96, 1.04, 1.1],
        shock_activity_history=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        structural_drift_history=[0.34, 0.4, 0.51, 0.58, 0.67, 0.74],
        trajectory_analysis=_trajectory(stabilizing=0.22, metastable=0.28, diverging=0.5),
        branching_analysis={"commitment": "MODERATE", "branch_tension": 0.45},
        constraint_analysis={"lock_in_score": 0.58, "recovery_margin": 0.41},
    )
    assert out["available"] is True
    assert out["risk_horizon"] == "NEAR_TERM"


def test_horizon_mid_term_for_mixed_branching() -> None:
    out = estimate_risk_horizon(
        transition_pressure_history=[0.61, 0.74, 0.81, 0.86, 0.82, 0.88],
        shock_activity_history=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        structural_drift_history=[0.42, 0.46, 0.5, 0.47, 0.52, 0.5],
        trajectory_analysis=_trajectory(stabilizing=0.32, metastable=0.37, diverging=0.31),
        branching_analysis={"commitment": "LOW", "branch_tension": 0.86},
        constraint_analysis={"lock_in_score": 0.46, "recovery_margin": 0.48},
    )
    assert out["available"] is True
    assert out["risk_horizon"] == "MID_TERM"


def test_horizon_longer_for_recoverable_low_risk() -> None:
    out = estimate_risk_horizon(
        transition_pressure_history=[0.8, 0.65, 0.54, 0.46, 0.39, 0.31],
        shock_activity_history=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        structural_drift_history=[0.62, 0.51, 0.43, 0.35, 0.26, 0.2],
        trajectory_analysis=_trajectory(stabilizing=0.63, metastable=0.22, diverging=0.15),
        branching_analysis={"commitment": "HIGH", "branch_tension": 0.2},
        constraint_analysis={"lock_in_score": 0.22, "recovery_margin": 0.76},
    )
    assert out["available"] is True
    assert out["risk_horizon"] == "LONGER_HORIZON"


def test_horizon_safe_fallback_on_missing_inputs() -> None:
    out = estimate_risk_horizon(
        transition_pressure_history=None,
        shock_activity_history=None,
        structural_drift_history=None,
        trajectory_analysis=None,
        branching_analysis=None,
        constraint_analysis=None,
    )
    assert out["available"] is False
    assert out["risk_horizon"] == "MID_TERM"
