from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.timeframe_alignment import (
    apply_timeframe_confidence_adjustment,
    build_timeframe_alignment_table,
    compute_action_agreement,
    compute_regime_agreement,
)


def _tf_df(index: pd.DatetimeIndex, regime: str, action: str, conf: float) -> pd.DataFrame:
    n = len(index)
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            DATE_COLUMN: index,
            "regime_label": [regime] * n,
            "action_posture": [action] * n,
            "confidence_score": [conf] * n,
            "fwd_ret_1d": rng.normal(0.0, 0.01, n),
            "fwd_ret_5d": rng.normal(0.0, 0.02, n),
            "fwd_ret_10d": rng.normal(0.0, 0.03, n),
            "spy_vol_10d": np.clip(rng.normal(0.7, 0.1, n), 0.1, 1.2),
        }
    )


def test_alignment_table_columns_exist():
    daily = _tf_df(pd.date_range("2024-01-01", periods=4, freq="D"), "stable_trend", "lean_long", 0.8)
    hourly = _tf_df(pd.date_range("2024-01-01 09:30", periods=8, freq="12h"), "stable_trend", "lean_long", 0.7)
    intraday = _tf_df(pd.date_range("2024-01-01 09:30", periods=20, freq="15min"), "fragile_rally", "lean_long", 0.6)

    aligned = build_timeframe_alignment_table(daily, hourly, intraday)

    required = {
        "regime_daily", "regime_1h", "regime_15m",
        "action_daily", "action_1h", "action_15m",
        "confidence_daily", "confidence_1h", "confidence_15m",
    }
    assert required.issubset(aligned.columns)


def test_regime_action_and_adjusted_confidence_exist():
    daily = _tf_df(pd.date_range("2024-01-01", periods=4, freq="D"), "stable_trend", "lean_long", 0.8)
    hourly = _tf_df(pd.date_range("2024-01-01 09:30", periods=8, freq="12h"), "fragile_rally", "reduce_exposure", 0.7)
    intraday = _tf_df(pd.date_range("2024-01-01 09:30", periods=20, freq="15min"), "high_volatility", "lean_long", 0.6)

    out = build_timeframe_alignment_table(daily, hourly, intraday)
    out = compute_regime_agreement(out)
    out = compute_action_agreement(out)
    out = apply_timeframe_confidence_adjustment(out)

    assert {"regime_agreement_score", "regime_alignment_label"}.issubset(out.columns)
    assert {"action_agreement_score", "action_alignment_label"}.issubset(out.columns)
    assert "adjusted_confidence_score" in out.columns
    assert out["adjusted_confidence_score"].between(0.0, 1.0).all()
    assert out[DATE_COLUMN].is_monotonic_increasing
    assert out.columns.duplicated().sum() == 0
