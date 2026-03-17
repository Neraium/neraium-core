from __future__ import annotations

import numpy as np

from neraium_core.spectral import (
    dominant_mode_loadings,
    eigendecomposition,
    spectral_gap,
    spectral_radius,
)


def test_spectral_observables() -> None:
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    eigenvalues, eigenvectors = eigendecomposition(matrix)
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)
    assert spectral_radius(matrix) >= 1.0
    assert spectral_gap(matrix) > 0.0


def test_dominant_mode_loadings_are_normalized() -> None:
    matrix = np.array([[1.0, 0.8, 0.1], [0.8, 1.0, 0.2], [0.1, 0.2, 1.0]])
    loadings = dominant_mode_loadings(matrix)
    assert loadings.shape == (3,)
    assert np.isclose(loadings.sum(), 1.0)
    assert np.all(loadings >= 0.0)
