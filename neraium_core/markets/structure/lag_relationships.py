"""Simple lead-lag structure approximations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lag_structure_shift(baseline: pd.DataFrame, recent: pd.DataFrame, lag: int = 1) -> float:
    base_shift = baseline.shift(lag).corrwith(baseline, axis=0).fillna(0.0)
    recent_shift = recent.shift(lag).corrwith(recent, axis=0).fillna(0.0)
    return float(np.mean(np.abs(base_shift - recent_shift)))
