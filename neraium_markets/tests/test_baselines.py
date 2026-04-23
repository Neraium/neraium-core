"""Day 5 tests for baseline construction and comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.baselines import (
    baseline_breadth_signal,
    baseline_trend_signal,
    baseline_vol_signal,
    compare_to_baselines,
)
from neraium.evaluation import compute_forward_returns


def _base_df(n: int = 70) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    spy = 100 + np.cumsum(rng.normal(0.0, 0.8, n))
    spy_ret_5d = pd.Series(spy).pct_change(5).to_numpy()
    return pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "spy": spy,
            "spy_ret_5d": spy_ret_5d,
            "spy_vol_10d": np.clip(rng.normal(0.7, 0.2, n), 0.0, 1.2),
            "breadth_pct_above_20dma": np.clip(rng.normal(50, 15, n), 0, 100),
            "action_posture": "watch",
        }
    )


def test_baseline_columns_exist():
    df = _base_df()
    df = baseline_trend_signal(df)
    df = baseline_vol_signal(df)
    df = baseline_breadth_signal(df)
    assert "baseline_trend_action" in df.columns
    assert "baseline_vol_action" in df.columns
    assert "baseline_breadth_action" in df.columns


def test_baseline_comparison_not_empty():
    df = compute_forward_returns(_base_df())
    cmp_df = compare_to_baselines(df)
    assert not cmp_df.empty
    assert {"model", "horizon", "avg_usefulness", "hit_rate"}.issubset(cmp_df.columns)
