from __future__ import annotations

import pandas as pd

from sii.spectral import spectral_observables


def test_spectral_observables() -> None:
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["x", "y"], index=["x", "y"])
    out = spectral_observables(corr)
    assert out["spectral_radius"] >= 1.0
    assert len(out["dominant_eigenvector"]) == 2
