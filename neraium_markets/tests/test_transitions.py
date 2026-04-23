"""Day 6: regime runs and transition matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import compute_forward_returns, score_action_usefulness
from neraium.transitions import (
    build_transition_matrix,
    compute_regime_runs,
    summarize_regime_persistence,
    summarize_transition_quality,
)


def _base_df() -> pd.DataFrame:
    n = 20
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    regimes = ["a", "a", "a", "b", "b", "c", "c", "c", "c", "a"] + ["a"] * (n - 10)
    return pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "regime_label": regimes,
            "action_posture": ["watch"] * n,
            "confidence_score": np.linspace(0.4, 0.9, n),
            "spy": 100 + np.arange(n) * 0.1,
            "spy_vol_10d": [0.7] * n,
        }
    )


def test_regime_run_columns_exist():
    df = compute_regime_runs(_base_df())
    assert "regime_run_id" in df.columns
    assert "regime_run_length" in df.columns
    assert "regime_position_in_run" in df.columns


def test_persistence_summary_not_empty():
    df = compute_regime_runs(_base_df())
    s = summarize_regime_persistence(df)
    assert not s.empty
    assert "count_runs" in s.columns
    assert "pct_single_step_runs" in s.columns


def test_transition_matrix_not_empty():
    df = compute_regime_runs(_base_df())
    m = build_transition_matrix(df)
    assert not m.empty
    assert {"from_regime", "to_regime", "transition_count", "p_to_given_from"}.issubset(m.columns)


def test_transition_quality_not_empty():
    df = compute_forward_returns(_base_df())
    df = score_action_usefulness(df)
    df = compute_regime_runs(df)
    tq = summarize_transition_quality(df)
    assert not tq.empty
    assert "transition_count" in tq.columns
