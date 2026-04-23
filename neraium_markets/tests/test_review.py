"""Tests for Day 10 review tables."""

from __future__ import annotations

import pandas as pd

from neraium.decisions import attach_outcomes, generate_decision_record
from neraium.review import build_decision_review_table, identify_top_failures, identify_top_successes


def _full_df() -> pd.DataFrame:
    """Enough rows for rolling + guaranteed top failure/success."""
    rows = []
    ts = pd.date_range("2024-01-01", periods=30, freq="D")
    for i, t in enumerate(ts):
        rows.append(
            {
                "timestamp": t,
                "market_regime_label": "market_stable",
                "market_action_posture": "risk_on",
                "path_adjusted_market_action": "risk_on",
                "market_confidence_score": 0.75,
                "market_warning_level": "none",
                "scenario_path_label": "unclassified",
                "trajectory_direction": "flat",
                "path_aware_market_explanation": "explanation",
                "fwd_ret_1d": 0.001,
                "fwd_ret_5d": -0.02 if i < 3 else 0.02,
                "fwd_ret_10d": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    df = generate_decision_record(df)
    return attach_outcomes(df)


def test_review_table_columns_and_sorted() -> None:
    df = _full_df()
    rev = build_decision_review_table(df)
    expected = {
        "timestamp",
        "regime_label",
        "scenario_path_label",
        "trajectory_direction",
        "warning_level",
        "action_posture",
        "path_adjusted_market_action",
        "confidence_score",
        "explanation",
        "fwd_ret_5d",
        "action_useful_5d",
        "decision_quality_flag",
    }
    assert set(rev.columns) == expected
    assert rev["timestamp"].is_monotonic_increasing
    assert not rev.columns.duplicated().any()


def test_top_failures_and_successes_nonempty() -> None:
    df = _full_df()
    fails = identify_top_failures(df, n=3)
    succ = identify_top_successes(df, n=3)
    assert not fails.empty
    assert not succ.empty
    assert (fails["action_useful_5d"] == -1).all()
    assert (succ["action_useful_5d"] == 1).all()
