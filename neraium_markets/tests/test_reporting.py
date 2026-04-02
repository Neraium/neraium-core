"""Day 5 tests for validation report construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import compute_forward_returns, score_action_usefulness
from neraium.reporting import build_validation_report


def _report_df(n: int = 75) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    spy = 100 + np.cumsum(rng.normal(0.1, 0.9, n))
    spy_ret_5d = pd.Series(spy).pct_change(5).to_numpy()
    actions = np.array(["lean_long", "reduce_exposure", "wait", "watch", "avoid_risk"])
    regimes = np.array(["stable_trend", "fragile_rally", "mean_reversion", "high_volatility", "risk_off_transition"])

    return pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "spy": spy,
            "spy_ret_5d": spy_ret_5d,
            "spy_vol_10d": np.clip(rng.normal(0.7, 0.15, n), 0.0, 1.2),
            "breadth_pct_above_20dma": np.clip(rng.normal(50, 12, n), 0, 100),
            "action_posture": actions[np.arange(n) % len(actions)],
            "regime_label": regimes[np.arange(n) % len(regimes)],
            "confidence_score": np.clip(rng.uniform(0.0, 1.0, n), 0.0, 1.0),
        }
    )


def test_validation_report_has_required_keys():
    df = score_action_usefulness(compute_forward_returns(_report_df()))
    report = build_validation_report(df)

    required = {
        "total_signals",
        "regime_counts",
        "action_counts",
        "average_confidence",
        "usefulness_summary",
    }
    assert required.issubset(report.keys())
