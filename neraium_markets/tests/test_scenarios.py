"""Tests for Day 9 scenario path labeling."""

from __future__ import annotations

import pandas as pd

from neraium.scenarios import label_scenario_paths, summarize_scenario_paths


def test_label_scenario_paths_smoke() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "market_regime_label": [
                "market_stable",
                "market_stable",
                "market_fragile",
                "market_fragile",
                "market_risk_off",
            ],
            "market_action_posture": ["risk_on"] * 5,
            "fwd_ret_1d": [0.0] * 5,
            "fwd_ret_5d": [0.0] * 5,
            "fwd_ret_10d": [0.0] * 5,
            "market_run_length": [1, 2, 1, 2, 1],
        }
    )
    out = label_scenario_paths(df)
    assert "scenario_path_label" in out.columns
    assert "scenario_transition_flag" in out.columns
    assert "fragile_to_risk_off" in set(out["scenario_path_label"].astype(str))


def test_summarize_scenario_paths() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="D"),
            "scenario_path_label": ["unclassified", "stable_to_fragile", "stable_to_fragile", "unclassified"],
            "market_action_posture": ["risk_on", "neutral", "neutral", "wait"],
            "fwd_ret_1d": [0.001, -0.002, 0.0, 0.0],
            "fwd_ret_5d": [0.0, 0.0, 0.0, 0.0],
            "fwd_ret_10d": [0.0, 0.0, 0.0, 0.0],
            "market_run_length": [1, 1, 2, 1],
        }
    )
    s = summarize_scenario_paths(df)
    assert "scenario_path_label" in s.columns
    assert "count" in s.columns
    assert "avg_usefulness_1d" in s.columns
