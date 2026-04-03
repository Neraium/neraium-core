from __future__ import annotations

from neraium_core.branching import derive_branching_analysis
from neraium_core.experimental_analytics.trajectory_analysis import classify_trajectory_path


def test_branching_commitment_drops_when_temporal_consistency_is_low() -> None:
    trajectory = {
        "path_scores": {"stabilizing": 0.12, "metastable": 0.22, "diverging": 0.66},
        "diagnostics": {"directional_divergence_score": 0.25, "shape_divergence_score": 0.2},
    }
    high_q = derive_branching_analysis(
        trajectory,
        temporal_quality={"temporal_consistency_score": 0.95, "timestamp_gap_irregularity": 0.05},
        temporal_features={"timing_instability": 0.05},
    )
    low_q = derive_branching_analysis(
        trajectory,
        temporal_quality={"temporal_consistency_score": 0.2, "timestamp_gap_irregularity": 0.9},
        temporal_features={"timing_instability": 0.85},
    )
    assert high_q is not None and low_q is not None
    assert high_q["commitment"] in {"HIGH", "MODERATE"}
    assert low_q["branch_tension"] > high_q["branch_tension"]


def test_trajectory_path_reflects_timing_velocity_and_irregularity() -> None:
    base = dict(
        drift_history=[0.3, 0.42, 0.58, 0.7, 0.86, 1.0],
        transition_pressure_history=[0.45, 0.58, 0.74, 0.9, 1.08, 1.22],
        subsystem_instability_history=[0.25, 0.34, 0.48, 0.62, 0.78, 0.95],
        regime_novelty_history=[0.1, 0.14, 0.22, 0.32, 0.45, 0.62],
        shock_activity_history=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
        reversibility_classification="METASTABLE",
        reversibility_score=0.56,
    )
    smooth = classify_trajectory_path(
        **base,
        temporal_quality={"temporal_consistency_score": 0.95, "timestamp_gap_irregularity": 0.05},
        temporal_features={"drift_velocity": 0.2, "drift_acceleration": 0.15, "timing_instability": 0.1},
    )
    unstable = classify_trajectory_path(
        **base,
        temporal_quality={"temporal_consistency_score": 0.25, "timestamp_gap_irregularity": 0.85},
        temporal_features={"drift_velocity": 0.8, "drift_acceleration": 0.7, "timing_instability": 0.9},
    )
    assert unstable["path_scores"]["diverging"] >= smooth["path_scores"]["diverging"]
