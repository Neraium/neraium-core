from __future__ import annotations

from neraium_core.experimental_analytics.trajectory_analysis import classify_trajectory_path


def test_trajectory_analysis_stable_recovery_is_stabilizing() -> None:
    out = classify_trajectory_path(
        drift_history=[0.72, 0.66, 0.55, 0.44, 0.31, 0.22],
        transition_pressure_history=[1.05, 0.98, 0.83, 0.68, 0.54, 0.41],
        subsystem_instability_history=[1.1, 1.0, 0.86, 0.72, 0.58, 0.43],
        regime_novelty_history=[0.35, 0.32, 0.27, 0.22, 0.16, 0.1],
        shock_activity_history=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        reversibility_classification="REVERSIBLE",
        reversibility_score=0.18,
    )
    assert out["dominant_path"] == "STABILIZING"
    scores = out["path_scores"]
    assert scores["stabilizing"] > scores["diverging"]
    assert scores["stabilizing"] > scores["metastable"]


def test_trajectory_analysis_escalation_is_diverging() -> None:
    out = classify_trajectory_path(
        drift_history=[0.22, 0.29, 0.36, 0.52, 0.71, 0.96],
        transition_pressure_history=[0.34, 0.45, 0.6, 0.82, 1.02, 1.22],
        subsystem_instability_history=[0.28, 0.35, 0.43, 0.63, 0.92, 1.2],
        regime_novelty_history=[0.08, 0.13, 0.21, 0.35, 0.5, 0.69],
        shock_activity_history=[0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        reversibility_classification="LOCKED_IN",
        reversibility_score=0.86,
    )
    assert out["dominant_path"] == "DIVERGING"
    scores = out["path_scores"]
    assert scores["diverging"] > scores["stabilizing"]
    assert scores["diverging"] > scores["metastable"]


def test_trajectory_analysis_oscillating_fragile_is_metastable() -> None:
    out = classify_trajectory_path(
        drift_history=[0.48, 0.66, 0.5, 0.68, 0.52, 0.65],
        transition_pressure_history=[0.74, 0.97, 0.79, 1.01, 0.83, 0.99],
        subsystem_instability_history=[0.74, 0.98, 0.77, 1.02, 0.8, 1.0],
        regime_novelty_history=[0.35, 0.42, 0.39, 0.47, 0.44, 0.48],
        shock_activity_history=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        reversibility_classification="METASTABLE",
        reversibility_score=0.51,
    )
    assert out["dominant_path"] == "METASTABLE"
    scores = out["path_scores"]
    assert scores["metastable"] > scores["stabilizing"]
    assert scores["metastable"] > scores["diverging"]
