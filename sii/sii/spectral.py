from __future__ import annotations

import numpy as np
import pandas as pd


def spectral_observables(corr: pd.DataFrame) -> dict[str, object]:
    vals, vecs = np.linalg.eigh(corr.values)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    spectral_radius = float(vals[0])
    spectral_gap = float(vals[0] - vals[1]) if len(vals) > 1 else 0.0
    dominant = vecs[:, 0]
    ranked = sorted(
        zip(corr.columns, np.abs(dominant), strict=False),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "spectral_radius": spectral_radius,
        "spectral_gap": spectral_gap,
        "dominant_eigenvector": dominant.tolist(),
        "ranked_signal_loadings": [(k, float(v)) for k, v in ranked],
        "eigenvalues": vals.tolist(),
    }
