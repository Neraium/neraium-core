"""Tests for Day 9 warnings and path utilities."""

from __future__ import annotations

import pandas as pd

from neraium.warnings import (
    adjust_market_action_for_path,
    compare_path_aware_vs_static_market_usefulness,
    compute_early_warning_flags,
    compute_path_persistence_score,
    compute_reversal_risk_score,
    generate_market_warning_level,
    refine_early_warnings_with_reversal,
)


def _base() -> pd.DataFrame:
    n = 8
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "market_regime_label": ["market_stable"] * 3 + ["market_fragmented"] * 5,
            "market_confidence_score": [0.6] * n,
            "market_instability_score": [0.4, 0.42, 0.44, 0.5, 0.52, 0.54, 0.56, 0.58],
            "market_coherence_score": [0.55, 0.54, 0.53, 0.5, 0.48, 0.46, 0.44, 0.42],
            "instability_delta": [0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "coherence_delta": [0.0, -0.01, -0.01, -0.02, -0.02, -0.02, -0.02, -0.02],
            "confidence_delta": [0.0] * n,
            "trajectory_speed_score": [0.1] * 5 + [0.5, 0.55, 0.6],
            "trajectory_direction": ["flat", "improving", "deteriorating", "deteriorating", "reversing", "flat", "flat", "flat"],
            "scenario_path_label": ["unclassified"] * n,
            "regime_change_flag": [0, 0, 0, 1, 0, 0, 0, 0],
            "market_position_in_run": [1, 2, 3, 1, 2, 3, 4, 5],
            "market_run_length": [3, 3, 3, 5, 5, 5, 5, 5],
            "trajectory_run_length": [1] * n,
            "scenario_path_length": [1] * n,
            "market_action_posture": ["risk_on"] * n,
        }
    )


def test_warning_pipeline_smoke() -> None:
    df = _base()
    df = compute_path_persistence_score(df)
    df = compute_reversal_risk_score(df)
    df = compute_early_warning_flags(df)
    df = refine_early_warnings_with_reversal(df)
    df = generate_market_warning_level(df)
    assert "warning_score" in df.columns
    assert "market_warning_level" in df.columns
    assert df["instability_rising_warning"].dtype in (int, "int64", "int32")


def test_adjust_and_compare() -> None:
    df = _base()
    df = compute_path_persistence_score(df)
    df = compute_reversal_risk_score(df)
    df = compute_early_warning_flags(df)
    df = refine_early_warnings_with_reversal(df)
    df = generate_market_warning_level(df)
    df = adjust_market_action_for_path(df)
    assert "path_adjusted_market_action" in df.columns
    df["fwd_ret_1d"] = 0.001
    df["fwd_ret_5d"] = 0.0
    df["fwd_ret_10d"] = 0.0
    cmp_df = compare_path_aware_vs_static_market_usefulness(df)
    assert len(cmp_df) == 2
    assert set(cmp_df["version"]) == {"static_market_action", "path_adjusted_market_action"}
