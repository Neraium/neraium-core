from __future__ import annotations

import numpy as np
import pandas as pd


def lagged_directional_matrix(window: pd.DataFrame) -> pd.DataFrame:
    cols = window.columns
    mat = np.zeros((len(cols), len(cols)), dtype=float)
    current = window.iloc[:-1]
    future = window.iloc[1:]
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            mat[i, j] = np.corrcoef(current[ci], future[cj])[0, 1]
    return pd.DataFrame(np.nan_to_num(mat), index=cols, columns=cols)


def directional_metrics(matrix: pd.DataFrame) -> dict[str, float]:
    abs_m = np.abs(matrix.values)
    energy = float(np.mean(abs_m))
    asymmetry = float(np.mean(np.abs(matrix.values - matrix.values.T)))
    divergence = float(energy * (1 + asymmetry))
    return {
        "causal_energy": energy,
        "causal_asymmetry": asymmetry,
        "causal_divergence": divergence,
    }
