"""Structural drift score aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neraium_core.markets.structure.concentration_entropy import concentration_entropy_shift
from neraium_core.markets.structure.correlation_geometry import correlation_distance
from neraium_core.markets.structure.lag_relationships import lag_structure_shift


def _squash(x: float, scale: float = 1.0) -> float:
    return float(1.0 - np.exp(-max(x, 0.0) / max(scale, 1e-6)))


def compute_drift_components(baseline: pd.DataFrame, recent: pd.DataFrame) -> dict[str, float]:
    corr = correlation_distance(baseline, recent)
    lag = lag_structure_shift(baseline, recent)
    conc, ent = concentration_entropy_shift(baseline, recent)
    return {
        "correlation_geometry": _squash(corr, 0.05),
        "lag_relationships": _squash(lag, 0.2),
        "concentration_shift": _squash(conc, 0.1),
        "entropy_shift": _squash(ent, 0.2),
    }


def compute_structural_drift_score(components: dict[str, float]) -> float:
    weights = {
        "correlation_geometry": 0.35,
        "lag_relationships": 0.2,
        "concentration_shift": 0.2,
        "entropy_shift": 0.25,
    }
    score = sum(components[k] * w for k, w in weights.items())
    return float(np.clip(score, 0.0, 1.0))


def compute_instability_score(state: pd.DataFrame) -> float:
    vol_cols = [c for c in state.columns if c.startswith("vol_")]
    if not vol_cols:
        return 0.0
    vol = state[vol_cols].mean(axis=1)
    slope = vol.tail(8).diff().mean()
    spread = vol.tail(20).std()
    raw = max(slope, 0.0) + spread
    return float(np.clip(raw * 8.0, 0.0, 1.0))
