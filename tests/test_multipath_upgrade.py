from __future__ import annotations

import numpy as np

from neraium_core.branching import derive_branching_analysis
from neraium_core.experimental_analytics.directional_evolution import derive_directional_evolution_features
from neraium_core.experimental_analytics.path_prototypes import derive_path_prototypes
from neraium_core.experimental_analytics.trajectory_analysis import classify_trajectory_path
from neraium_core.experimental_analytics.trajectory_shape_features import derive_trajectory_shape_features


def _trajectory(window: np.ndarray) -> dict[str, object]:
    directional = derive_directional_evolution_features(recent_window=window)
    shape = derive_trajectory_shape_features(recent_window=window)
    proto = derive_path_prototypes(directional_evolution=directional, trajectory_shape=shape)
    return classify_trajectory_path(
        drift_history=[0.28, 0.35, 0.42, 0.51, 0.59, 0.67],
        transition_pressure_history=[0.45, 0.54, 0.62, 0.73, 0.84, 0.93],
        subsystem_instability_history=[0.42, 0.48, 0.57, 0.69, 0.81, 0.92],
        regime_novelty_history=[0.11, 0.16, 0.23, 0.31, 0.39, 0.46],
        shock_activity_history=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        reversibility_classification="METASTABLE",
        reversibility_score=0.48,
        directional_evolution=directional,
        trajectory_shape=shape,
        path_prototypes=proto,
    )


def test_same_severity_different_direction_yields_distinct_candidates() -> None:
    t = np.linspace(0.0, 1.0, 12)
    # Same endpoint magnitude but opposite directional composition.
    window_a = np.stack([t, t * 0.8, np.sin(t * np.pi) * 0.3 + t * 0.1], axis=1)
    window_b = np.stack([1.0 - t, t * 0.8, np.cos(t * np.pi) * 0.3 - t * 0.1], axis=1)

    traj_a = _trajectory(window_a)
    traj_b = _trajectory(window_b)

    dir_a = traj_a["directional_evolution"]["normalized_first_difference_vector"]
    dir_b = traj_b["directional_evolution"]["normalized_first_difference_vector"]
    assert dir_a != dir_b
    assert traj_a["directional_evolution"]["directional_polarity_score"] != traj_b["directional_evolution"][
        "directional_polarity_score"
    ]


def test_branching_exposes_multipath_diagnostics() -> None:
    t = np.linspace(0.0, 1.0, 12)
    window = np.stack([t, np.sin(5.0 * t) * 0.4 + t * 0.25, np.where(t > 0.55, 1.2 * t, 0.3 * t)], axis=1)
    trajectory = _trajectory(window)
    branching = derive_branching_analysis(trajectory)
    assert branching is not None
    assert branching["available"] is True
    assert branching["branch_count_estimate"] >= 2
    assert branching["diagnostics"]["path_prototype_count"] >= 2
    assert "candidate_paths" in branching
