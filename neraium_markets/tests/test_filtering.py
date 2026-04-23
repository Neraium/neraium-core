"""Day 6: false-positive flags and filtered signal comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.diagnostics import compute_signal_stability
from neraium.evaluation import compute_forward_returns, score_action_usefulness
from neraium.filtering import apply_signal_filters, compare_filtered_vs_unfiltered, identify_false_positive_patterns
from neraium.transitions import compute_regime_runs


def _full_chain() -> pd.DataFrame:
    n = 25
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "regime_label": ["mean_reversion"] * n,
            "action_posture": np.array(["lean_long", "wait", "watch", "reduce_exposure"] * 6 + ["lean_long"]),
            "confidence_score": np.clip(np.linspace(0.3, 0.95, n), 0, 1),
            "coherence_score": np.linspace(0.2, 0.8, n),
            "instability_score": np.linspace(0.2, 0.9, n),
            "corr_drift_score": np.linspace(0.1, 0.5, n),
            "breadth_pct_above_20dma": np.linspace(30, 70, n),
            "spy": 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, n)),
            "spy_vol_10d": [0.75] * n,
        }
    )
    df = compute_forward_returns(df)
    df = score_action_usefulness(df)
    df = compute_regime_runs(df)
    df = compute_signal_stability(df)
    df = identify_false_positive_patterns(df)
    df = apply_signal_filters(df)
    df = score_action_usefulness(df, action_col="filtered_action_posture", out_prefix="filtered_useful")
    return df


def test_false_positive_flags_exist():
    df = _full_chain()
    for c in ("low_persistence_flag", "contradiction_flag", "unstable_signal_flag", "likely_false_positive_flag"):
        assert c in df.columns


def test_filtered_action_exists():
    df = _full_chain()
    assert "filtered_action_posture" in df.columns
    assert "signal_kept_flag" in df.columns


def test_filtered_comparison_not_empty():
    df = _full_chain()
    cmp_df = compare_filtered_vs_unfiltered(df)
    assert not cmp_df.empty
    assert (cmp_df["version"] == "unfiltered").any()
    assert (cmp_df["version"] == "filtered").any()


def test_sorted_output_preserved():
    df = _full_chain()
    assert df[DATE_COLUMN].is_monotonic_increasing


def test_no_duplicate_columns():
    df = _full_chain()
    assert df.columns.is_unique
