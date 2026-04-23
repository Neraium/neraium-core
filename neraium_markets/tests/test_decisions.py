"""Tests for Day 10 decision records and quality."""

from __future__ import annotations

import pandas as pd

from neraium.decisions import (
    attach_outcomes,
    build_decision_memory,
    decision_quality_summary_table,
    evaluate_decision_quality,
    generate_decision_record,
)


def _synthetic_market_state() -> pd.DataFrame:
    n = 25
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "market_regime_label": ["market_stable"] * n,
            "market_action_posture": ["risk_on"] * n,
            "path_adjusted_market_action": ["risk_on"] * n,
            "market_confidence_score": [0.4] * 5 + [0.7] * 15 + [0.6] * 5,
            "market_warning_level": ["none"] * n,
            "scenario_path_label": ["unclassified"] * n,
            "trajectory_direction": ["flat"] * n,
            "fwd_ret_1d": [0.01] * n,
            "fwd_ret_5d": [-0.02] * 8 + [0.03] * 10 + [0.0] * 7,
            "fwd_ret_10d": [0.0] * n,
        }
    )


def test_generate_decision_record_has_decision_type() -> None:
    df = generate_decision_record(_synthetic_market_state())
    assert "decision_type" in df.columns
    assert (df["decision_type"] == "act").all()


def test_attach_outcomes_forward_returns_and_usefulness() -> None:
    base = generate_decision_record(_synthetic_market_state())
    out = attach_outcomes(base)
    assert "action_useful_1d" in out.columns
    assert "action_useful_5d" in out.columns
    assert "action_useful_10d" in out.columns
    assert not out.columns.duplicated().any()


def test_evaluate_decision_quality() -> None:
    base = generate_decision_record(_synthetic_market_state())
    out = attach_outcomes(base)
    q = evaluate_decision_quality(out)
    assert q["total_decisions"] == len(out)
    assert "act" in q["counts_by_decision_type"]
    tbl = decision_quality_summary_table(q)
    assert len(tbl) >= 6


def test_build_decision_memory_rolling() -> None:
    base = generate_decision_record(_synthetic_market_state())
    out = attach_outcomes(base)
    mem = build_decision_memory(out, window=10)
    assert "rolling_usefulness" in mem.columns
    assert "rolling_success_rate" in mem.columns
    assert "rolling_confidence_accuracy" in mem.columns
    assert mem["timestamp"].is_monotonic_increasing
