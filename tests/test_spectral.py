from __future__ import annotations

import numpy as np

from neraium_core.spectral import eigendecomposition, spectral_gap, spectral_radius


def test_spectral_observables() -> None:
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    eigenvalues, eigenvectors = eigendecomposition(matrix)
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)
    assert spectral_radius(matrix) >= 1.0
    assert spectral_gap(matrix) > 0.0
