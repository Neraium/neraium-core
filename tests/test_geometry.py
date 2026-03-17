from __future__ import annotations

import numpy as np

from neraium_core.geometry import (
    correlation_matrix,
    normalize_window,
    relational_structure,
    signal_structural_importance,
    structural_drift,
)


def test_geometry_correlation_and_structure() -> None:
    observations = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 5.0, 8.0], [4.0, 7.0, 11.0]])
    corr = correlation_matrix(observations)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)

    structure = relational_structure(corr)
    assert structure["centrality"].shape == (3,)
    assert structure["relational_energy"] >= 0.0


def test_normalize_window_and_structural_drift() -> None:
    window = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0]])
    z, mean, std = normalize_window(window)
    assert z.shape == window.shape
    assert np.allclose(mean, np.array([2.0, 12.0]))
    assert np.all(std > 0)

    base_corr = correlation_matrix(z)
    shifted = z + np.array([[0.0, 0.0], [0.2, -0.1], [0.1, -0.2]])
    curr_corr = correlation_matrix(shifted)
    assert structural_drift(curr_corr, base_corr, norm="fro") >= 0.0
    assert structural_drift(curr_corr, base_corr, norm="mae") >= 0.0


def test_signal_structural_importance() -> None:
    corr = np.array([[1.0, 0.4], [0.4, 1.0]])
    importance = signal_structural_importance(corr)
    assert importance.shape == (2,)
    assert np.all(importance >= 0.0)
