"""Tests for regime classification."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.regime import REGIME_LABELS, classify_regime


def _minimal_structural(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "instability_score": rng.uniform(0.1, 0.9, n),
            "corr_drift_score": rng.uniform(0.1, 0.9, n),
            "lag_drift_score": rng.uniform(0.1, 0.9, n),
            "coherence_score": rng.uniform(0.1, 0.9, n),
            "risk_off_proxy": rng.normal(0, 1, n),
            "breadth_pct_above_20dma": rng.uniform(30, 80, n),
            "spy_vol_20d": rng.uniform(0.01, 0.05, n),
        }
    )
    return base


def test_regime_label_exists():
    df = classify_regime(_minimal_structural())
    assert "regime_label" in df.columns


def test_regime_values_are_valid():
    df = classify_regime(_minimal_structural(80))
    for v in df["regime_label"].unique():
        assert v in REGIME_LABELS
