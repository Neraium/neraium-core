from __future__ import annotations

import numpy as np

from neraium_core.geometry import correlation_matrix, relational_structure


def test_geometry_correlation_and_structure() -> None:
    observations = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 5.0], [3.0, 5.0, 8.0], [4.0, 7.0, 11.0]])
    corr = correlation_matrix(observations)
    assert corr.shape == (3, 3)
    assert np.allclose(np.diag(corr), 1.0)

    structure = relational_structure(corr)
    assert structure["centrality"].shape == (3,)
    assert structure["relational_energy"] >= 0.0
