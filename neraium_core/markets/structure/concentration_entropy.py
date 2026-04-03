"""Concentration and entropy structure metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def concentration_index(window: pd.DataFrame) -> float:
    vol = window.std().abs()
    weights = vol / max(vol.sum(), 1e-9)
    return float(np.square(weights).sum())


def disorder_entropy(window: pd.DataFrame) -> float:
    vol = window.std().abs()
    weights = (vol / max(vol.sum(), 1e-9)).to_numpy() + 1e-12
    return float(-np.sum(weights * np.log(weights)))


def concentration_entropy_shift(baseline: pd.DataFrame, recent: pd.DataFrame) -> tuple[float, float]:
    conc_delta = abs(concentration_index(recent) - concentration_index(baseline))
    entropy_delta = abs(disorder_entropy(recent) - disorder_entropy(baseline))
    return float(conc_delta), float(entropy_delta)
