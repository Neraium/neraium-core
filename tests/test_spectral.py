from __future__ import annotations

import numpy as np

from neraium_core.spectral import (
    dominant_mode_loading,
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

    dominant = dominant_mode_loading(matrix)
    assert dominant["dominant_eigenvalue"] >= 1.0
    assert len(dominant["dominant_eigenvector"]) == 2
