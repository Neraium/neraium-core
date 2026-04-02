"""Day 5 tests for forward evaluation and calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.evaluation import (
    compute_forward_returns,
    evaluate_confidence_calibration,
    score_action_usefulness,
)


def _signals_like(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    spy = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    actions = np.array(["lean_long", "lean_short", "reduce_exposure", "avoid_risk", "wait", "watch"])
    return pd.DataFrame(
        {
            DATE_COLUMN: idx,
            "spy": spy,
            "spy_vol_10d": np.clip(rng.normal(0.7, 0.2, n), 0.0, 1.2),
            "action_posture": actions[np.arange(n) % len(actions)],
            "confidence_score": np.clip(rng.uniform(0, 1, n), 0.0, 1.0),
            "regime_label": "mean_reversion",
        }
    )


def test_forward_return_columns_exist():
    df = compute_forward_returns(_signals_like())
    for c in ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d"]:
        assert c in df.columns


def test_action_usefulness_columns_exist():
    df = compute_forward_returns(_signals_like())
    scored = score_action_usefulness(df)
    for c in ["action_useful_1d", "action_useful_5d", "action_useful_10d"]:
        assert c in scored.columns


def test_calibration_output_not_empty():
    df = compute_forward_returns(_signals_like())
    scored = score_action_usefulness(df)
    calibration = evaluate_confidence_calibration(scored)
    assert not calibration.empty
    assert (calibration["count"] >= 0).all()


def test_outputs_sorted():
    df = _signals_like().sample(frac=1.0, random_state=3).reset_index(drop=True)
    scored = score_action_usefulness(compute_forward_returns(df))
    assert scored[DATE_COLUMN].is_monotonic_increasing


def test_no_duplicate_columns():
    df = score_action_usefulness(compute_forward_returns(_signals_like()))
    assert df.columns.is_unique
