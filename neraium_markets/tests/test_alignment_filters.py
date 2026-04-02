from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.alignment_filters import (
    apply_timeframe_alignment_filter,
    compare_alignment_filtered_vs_unfiltered,
)
from neraium.timeframe_alignment import (
    apply_timeframe_confidence_adjustment,
    build_timeframe_alignment_table,
    compute_action_agreement,
    compute_regime_agreement,
)


def _make_df() -> pd.DataFrame:
    n = 30
    idx = pd.date_range("2024-01-01 09:30", periods=n, freq="15min")
    daily = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01", periods=3, freq="D"),
            "regime_label": ["risk_off_transition", "risk_off_transition", "high_volatility"],
            "action_posture": ["avoid_risk", "reduce_exposure", "avoid_risk"],
            "confidence_score": [0.7, 0.75, 0.8],
        }
    )
    hourly = pd.DataFrame(
        {
            DATE_COLUMN: pd.date_range("2024-01-01 09:30", periods=8, freq="1h"),
            "regime_label": ["high_volatility"] * 8,
            "action_posture": ["reduce_exposure"] * 8,
            "confidence_score": [0.65] * 8,
        }
    )
    intraday = pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "regime_label": np.where(np.arange(n) % 2 == 0, "fragile_rally", "unstable"),
            "action_posture": ["lean_long"] * n,
            "confidence_score": np.clip(np.linspace(0.35, 0.7, n), 0, 1),
            "fwd_ret_1d": np.linspace(-0.01, 0.01, n),
            "fwd_ret_5d": np.linspace(-0.02, 0.02, n),
            "fwd_ret_10d": np.linspace(-0.03, 0.03, n),
            "spy_vol_10d": np.clip(np.linspace(0.6, 0.95, n), 0, 1.5),
        }
    )

    out = build_timeframe_alignment_table(daily, hourly, intraday)
    out = compute_regime_agreement(out)
    out = compute_action_agreement(out)
    out = apply_timeframe_confidence_adjustment(out)
    return out


def test_alignment_filter_outputs_exist():
    out = apply_timeframe_alignment_filter(_make_df())
    assert {"alignment_filter_flag", "aligned_action_posture", "alignment_kept_flag"}.issubset(out.columns)


def test_alignment_comparison_not_empty():
    out = apply_timeframe_alignment_filter(_make_df())
    comp = compare_alignment_filtered_vs_unfiltered(out)
    assert not comp.empty
    assert set(comp["version"]) == {"unaligned", "alignment_filtered"}


def test_sorted_output_preserved_and_no_duplicate_columns():
    out = apply_timeframe_alignment_filter(_make_df())
    assert out[DATE_COLUMN].is_monotonic_increasing
    assert out.columns.duplicated().sum() == 0
