from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_geometry(z_window: pd.DataFrame) -> pd.DataFrame:
    corr = z_window.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def structural_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[float, float]:
    delta = current.values - baseline.values
    return float(np.linalg.norm(delta, ord="fro")), float(np.mean(np.abs(delta)))


def structural_centrality(corr: pd.DataFrame) -> dict[str, float]:
    values = np.mean(np.abs(corr.values), axis=1)
    return dict(zip(corr.columns, values, strict=False))
