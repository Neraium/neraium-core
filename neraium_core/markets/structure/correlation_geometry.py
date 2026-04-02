"""Correlation geometry calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_distance(baseline: pd.DataFrame, recent: pd.DataFrame) -> float:
    c1 = baseline.corr().fillna(0.0).to_numpy()
    c2 = recent.corr().fillna(0.0).to_numpy()
    return float(np.linalg.norm(c1 - c2) / max(c1.size, 1))
