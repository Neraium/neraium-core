from __future__ import annotations

import numpy as np

from neraium_core.geometry import correlation_matrix, relational_drift, relational_structure


def test_geometry_correlation_and_structure() -> None:
    observations = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 5.0, 8.0], [4.0, 7.0, 11.0]])
    corr = correlation_matrix(observations)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)

    structure = relational_structure(corr)
    assert structure["centrality"].shape == (3,)
    assert structure["relational_energy"] >= 0.0


def test_geometry_relational_drift_against_baseline() -> None:
    baseline = np.array([[1.0, 0.4], [0.4, 1.0]])
    current = np.array([[1.0, -0.2], [-0.2, 1.0]])
    drift = relational_drift(current, baseline)

    assert drift["frobenius_drift"] > 0.0
    assert drift["relative_drift"] > 0.0
    assert np.isclose(drift["max_abs_drift"], 0.6)
