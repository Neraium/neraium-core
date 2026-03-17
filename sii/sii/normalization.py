from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_window(window: pd.DataFrame) -> pd.DataFrame:
    mu = window.mean(axis=0)
    sigma = window.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = (window - mu) / sigma
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
