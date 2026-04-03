from __future__ import annotations

from neraium_core.alignment import StructuralEngine
from neraium_core.experimental_analytics.constraint_analysis import analyze_constraint_lock_in


def test_constraint_analysis_recovering_scenario_favors_recovery_margin() -> None:
    out = analyze_constraint_lock_in(
        transition_pressure_history=[1.0, 0.9, 0.75, 0.62, 0.48, 0.35],
        shock_activity_history=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        structural_drift_score=0.24,
        regime_novelty=0.12,
        regime_distance=0.08,
        subsystem_instability=0.34,
        reversibility_classification="REVERSIBLE",
        reversibility_score=0.18,
        trajectory_analysis={"path_scores": {"stabilizing": 0.61, "metastable": 0.23, "diverging": 0.16}},
    )

    assert out["available"] is True
    assert out["recovery_margin"] > out["lock_in_score"]
    assert out["commitment_to_failure"] == "LOW"
    assert out["point_of_no_return_risk"] == "LOW"


def test_constraint_analysis_persistent_failure_has_high_lock_in() -> None:
    out = analyze_constraint_lock_in(
        transition_pressure_history=[0.62, 0.86, 1.0, 1.12, 1.21, 1.26],
        shock_activity_history=[0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        structural_drift_score=0.98,
        regime_novelty=0.79,
        regime_distance=1.08,
        subsystem_instability=1.23,
        reversibility_classification="LOCKED_IN",
        reversibility_score=0.87,
        trajectory_analysis={"path_scores": {"stabilizing": 0.1, "metastable": 0.19, "diverging": 0.71}},
    )

    assert out["available"] is True
    assert out["lock_in_score"] > 0.7
    assert out["lock_in_score"] > out["recovery_margin"]
    assert out["commitment_to_failure"] == "HIGH"
    assert out["point_of_no_return_risk"] in {"ELEVATED", "HIGH"}


def test_constraint_analysis_fragile_metastable_stays_moderate() -> None:
    out = analyze_constraint_lock_in(
        transition_pressure_history=[0.7, 0.94, 0.78, 0.99, 0.84, 1.01],
        shock_activity_history=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        structural_drift_score=0.57,
        regime_novelty=0.44,
        regime_distance=0.53,
        subsystem_instability=0.92,
        reversibility_classification="METASTABLE",
        reversibility_score=0.49,
        trajectory_analysis={"path_scores": {"stabilizing": 0.24, "metastable": 0.52, "diverging": 0.24}},
    )

    assert out["available"] is True
    assert 0.3 <= out["recovery_margin"] <= 0.7
    assert 0.3 <= out["lock_in_score"] <= 0.7
    assert out["commitment_to_failure"] == "MODERATE"
    assert out["point_of_no_return_risk"] == "ELEVATED"


def test_constraint_analysis_safe_fallback_when_relational_metrics_are_skipped() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)

    latest = None
    for index in range(10):
        latest = engine.process_frame(
            {
                "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
                "site_id": "site-a",
                "asset_id": "asset-1",
                "sensor_values": {"pressure": 60.0 + (index * 0.2)},
            }
        )

    assert latest is not None
    constraint = latest["experimental_analytics"]["constraint_analysis"]
    assert constraint["available"] is False
    assert constraint["reason"] == "relational_metrics_skipped"
    assert constraint["recovery_margin"] is None
    assert constraint["lock_in_score"] is None
