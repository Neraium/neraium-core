"""Tests for Day 4 signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.regime import ACTION_POSTURES
from neraium.signals import generate_signals
from neraium.structural import build_structural_snapshot


def _feature_like(n: int = 60) -> pd.DataFrame:
    """Synthetic aligned closes (uppercase tickers, same as ``align_close_series``)."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({DATE_COLUMN: idx})
    for sym in ["SPY", "QQQ", "IWM"]:
        px = 100 + np.cumsum(rng.normal(0, 0.5, n))
        df[sym] = px
    for sym in ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]:
        df[sym] = 50 + np.cumsum(rng.normal(0, 0.3, n))
    for sym in ["VIX", "DXY", "GOLD", "OIL", "US2Y", "US10Y"]:
        df[sym] = 20 + rng.uniform(-1, 1, n)
    return df


def test_signal_columns_and_validity():
    from neraium.features import build_feature_table

    raw = _feature_like(55)
    feats = build_feature_table(raw)
    snap = build_structural_snapshot(feats)
    sig = generate_signals(snap)

    for col in [
        "confidence_score",
        "action_posture",
        DATE_COLUMN,
        "regime_label",
        "explanation",
        "instability_score",
        "coherence_score",
    ]:
        assert col in sig.columns

    assert sig["confidence_score"].between(0, 1, inclusive="both").all()
    for p in sig["action_posture"].unique():
        assert p in ACTION_POSTURES
    assert sig["explanation"].astype(str).str.len().gt(0).all()


def test_no_duplicate_columns_sorted():
    from neraium.features import build_feature_table

    raw = _feature_like(40)
    feats = build_feature_table(raw)
    snap = build_structural_snapshot(feats)
    sig = generate_signals(snap)
    assert sig.columns.is_unique
    assert sig[DATE_COLUMN].is_monotonic_increasing
