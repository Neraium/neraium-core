from __future__ import annotations

from neraium_core.experimental_analytics.counterfactual_simulation import simulate_counterfactual_futures


def _analysis(stabilizing: float, metastable: float, diverging: float) -> dict[str, object]:
    return {
        "dominant_path": "DIVERGING",
        "path_scores": {
            "stabilizing": stabilizing,
            "metastable": metastable,
            "diverging": diverging,
        },
    }


def test_diverging_high_commitment_relief_improves_modestly() -> None:
    out = simulate_counterfactual_futures(
        transition_pressure_history=[0.8, 0.94, 1.02, 1.11, 1.19, 1.24],
        shock_activity_history=[0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        structural_drift_history=[0.42, 0.54, 0.67, 0.79, 0.92, 1.04],
        trajectory_analysis=_analysis(0.12, 0.21, 0.67),
        branching_analysis={"commitment": "HIGH", "branch_tension": 0.28, "top_two_margin": 0.46},
        constraint_analysis={"lock_in_score": 0.83, "recovery_margin": 0.22},
        horizon_analysis={"risk_horizon": "NEAR_TERM"},
    )

    assert out["available"] is True
    assert out["baseline_future"]["projected_path"] == "DIVERGING"
    relief = next(c for c in out["counterfactuals"] if c["scenario"] == "pressure_relief")
    assert relief["projected_path"] in {"DIVERGING", "METASTABLE"}
    assert relief["delta_vs_baseline"]["lock_in_shift"] in {"IMPROVED", "IMPROVED_STRONGLY", "UNCHANGED"}


def test_mixed_branching_baseline_metastable_stabilized_structure_improves() -> None:
    out = simulate_counterfactual_futures(
        transition_pressure_history=[0.75, 0.82, 0.86, 0.84, 0.88, 0.89],
        shock_activity_history=[0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        structural_drift_history=[0.52, 0.56, 0.54, 0.57, 0.55, 0.56],
        trajectory_analysis=_analysis(0.31, 0.37, 0.32),
        branching_analysis={"commitment": "LOW", "branch_tension": 0.84, "top_two_margin": 0.04},
        constraint_analysis={"lock_in_score": 0.46, "recovery_margin": 0.48},
        horizon_analysis={"risk_horizon": "MID_TERM"},
    )

    assert out["baseline_future"]["projected_path"] == "METASTABLE"
    stabilized = next(c for c in out["counterfactuals"] if c["scenario"] == "stabilized_structure")
    assert stabilized["projected_path"] in {"STABILIZING", "METASTABLE"}
    assert stabilized["delta_vs_baseline"]["path_shift"] in {"IMPROVED", "IMPROVED_STRONGLY", "UNCHANGED"}


def test_already_stabilizing_relief_does_not_jump_absurdly() -> None:
    out = simulate_counterfactual_futures(
        transition_pressure_history=[0.78, 0.66, 0.58, 0.46, 0.38, 0.32],
        shock_activity_history=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        structural_drift_history=[0.69, 0.58, 0.49, 0.37, 0.28, 0.2],
        trajectory_analysis=_analysis(0.69, 0.2, 0.11),
        branching_analysis={"commitment": "MODERATE", "branch_tension": 0.24, "top_two_margin": 0.52},
        constraint_analysis={"lock_in_score": 0.21, "recovery_margin": 0.76},
        horizon_analysis={"risk_horizon": "LONGER_HORIZON"},
    )

    assert out["baseline_future"]["projected_path"] == "STABILIZING"
    relief = next(c for c in out["counterfactuals"] if c["scenario"] == "pressure_relief")
    assert relief["projected_path"] == "STABILIZING"
    assert relief["delta_vs_baseline"]["path_shift"] == "UNCHANGED"


def test_continued_degradation_worsens_when_already_deteriorating() -> None:
    out = simulate_counterfactual_futures(
        transition_pressure_history=[0.9, 1.0, 1.08, 1.16, 1.24, 1.29],
        shock_activity_history=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        structural_drift_history=[0.55, 0.67, 0.75, 0.86, 0.98, 1.1],
        trajectory_analysis=_analysis(0.11, 0.19, 0.7),
        branching_analysis={"commitment": "HIGH", "branch_tension": 0.21, "top_two_margin": 0.51},
        constraint_analysis={"lock_in_score": 0.84, "recovery_margin": 0.18},
        horizon_analysis={"risk_horizon": "IMMINENT"},
    )

    degraded = next(c for c in out["counterfactuals"] if c["scenario"] == "continued_degradation")
    assert degraded["projected_commitment"] == "HIGH"
    assert degraded["projected_horizon"] == "IMMINENT"
    assert degraded["delta_vs_baseline"]["commitment_shift"] in {"UNCHANGED", "WORSENED", "WORSENED_STRONGLY"}


def test_missing_required_histories_returns_clean_fallback() -> None:
    out = simulate_counterfactual_futures(
        transition_pressure_history=None,
        shock_activity_history=None,
        structural_drift_history=None,
    )

    assert out["available"] is False
    assert out["reason"] == "insufficient_required_histories"
    assert out["counterfactuals"] == []
