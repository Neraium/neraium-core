"""Tests for Day 9 market state trajectories."""

from __future__ import annotations

import pandas as pd

from neraium.trajectories import compute_market_state_runs, compute_market_state_trajectory


def _tiny_market() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "market_regime_label": [
                "market_stable",
                "market_stable",
                "market_fragile",
                "market_fragile",
                "market_risk_off",
            ],
            "market_confidence_score": [0.6, 0.62, 0.55, 0.50, 0.45],
            "market_instability_score": [0.4, 0.41, 0.5, 0.55, 0.7],
            "market_coherence_score": [0.55, 0.54, 0.5, 0.45, 0.35],
        }
    )


def test_trajectory_has_deltas_and_direction() -> None:
    df = compute_market_state_trajectory(_tiny_market())
    assert "trajectory_direction" in df.columns
    assert "instability_delta" in df.columns
    assert "trajectory_speed_score" in df.columns
    assert df["regime_change_flag"].sum() >= 1


def test_runs_segment_regime_and_trajectory() -> None:
    base = compute_market_state_trajectory(_tiny_market())
    out = compute_market_state_runs(base)
    assert "market_run_id" in out.columns
    assert "market_run_length" in out.columns
    assert "trajectory_run_id" in out.columns
    assert out["market_run_length"].max() >= 2
