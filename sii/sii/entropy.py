from __future__ import annotations

import numpy as np
import pandas as pd


def structural_entropy(corr: pd.DataFrame) -> float:
    vals = np.abs(corr.values).flatten()
    total = vals.sum()
    if total <= 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())
