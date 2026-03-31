from __future__ import annotations

import numpy as np

from neraium_core.stat_geometry.state_statistics import compute_state_statistics


def test_local_volume_stays_positive_for_nearly_collapsed_state_window() -> None:
    path = [np.array([1.0, 1.0, 1.0]) + (1e-10 * i) for i in range(20)]

    stats = compute_state_statistics(path, window=12)

    assert stats["local_volume"] > 0.0


def test_local_volume_varies_over_time_with_state_changes() -> None:
    base = [np.array([1.0 + 0.001 * i, 0.1 * i, 0.5]) for i in range(20)]
    shifted = base + [np.array([2.0 + 0.2 * i, -1.0 + 0.4 * i, 3.0]) for i in range(10)]

    before = compute_state_statistics(base, window=12)
    after = compute_state_statistics(shifted, window=12)

    assert before["local_volume"] > 0.0
    assert after["local_volume"] > 0.0
    assert before["local_volume"] != after["local_volume"]


def test_state_statistics_skips_tiny_covariance_windows() -> None:
    short_path = [np.array([0.1 * i, -0.2 * i, 1.0]) for i in range(6)]
    stats = compute_state_statistics(short_path, window=4, min_covariance_samples=20)
    assert stats["covariance_trace"] == 0.0
    assert stats["local_volume"] == 0.0


def test_state_statistics_handles_mixed_dimension_history_deterministically() -> None:
    path = [
        np.array([0.10, 0.20, 0.30]),
        np.array([0.12, 0.22, 0.31, 9.99]),
        np.array([0.15, 0.25, 0.33, 8.88, -1.0]),
        np.array([0.18, 0.28, 0.36]),
        np.array([0.22, 0.31, 0.40, 0.5]),
        np.array([0.27, 0.35, 0.45]),
        np.array([0.33, 0.40, 0.51, 1.5]),
        np.array([0.38, 0.45, 0.57]),
        np.array([0.44, 0.52, 0.63, -0.7]),
        np.array([0.51, 0.60, 0.70]),
        np.array([0.59, 0.69, 0.78, 0.1]),
        np.array([0.68, 0.79, 0.87]),
    ]

    first = compute_state_statistics(path, window=8, min_covariance_samples=8)
    second = compute_state_statistics(path, window=8, min_covariance_samples=8)

    assert first == second
    assert set(first.keys()) == {
        "local_volume",
        "covariance_logdet",
        "local_density",
        "covariance_trace",
        "principal_direction_strength",
        "anisotropy",
        "state_contraction_score",
        "state_expansion_score",
        "geometric_concentration",
    }
    assert first["local_volume"] > 0.0
