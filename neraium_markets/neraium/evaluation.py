"""Day 5 evaluation utilities for forward outcomes, action usefulness, and calibration."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from config import DATE_COLUMN


def compute_forward_returns(
    df: pd.DataFrame,
    price_col: str = "spy",
    horizons: list[int] = [1, 5, 10],
) -> pd.DataFrame:
    """
    Add deterministic forward return columns from a price series.

    For each horizon ``h`` this writes:
    ``fwd_ret_{h}d = price.shift(-h) / price - 1``.

    The function preserves all existing columns, sorts by timestamp ascending,
    and de-duplicates any duplicate column names.
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    resolved_col = price_col
    if resolved_col not in out.columns:
        alt = price_col.upper()
        if alt in out.columns:
            resolved_col = alt
        else:
            raise KeyError(f"Missing price column: {price_col}")

    price = out[resolved_col].astype(float)
    for h in horizons:
        if h <= 0:
            raise ValueError("Forward return horizons must be positive integers")
        out[f"fwd_ret_{h}d"] = (price.shift(-h) / price) - 1.0

    return out.loc[:, ~out.columns.duplicated()]


def _score_single_action(action: str, forward_ret: float, vol_10d: float | None = None) -> int:
    """
    Score a single action against realized forward behavior.

    Scoring scale:
    - 1 useful
    - 0 neutral
    - -1 harmful

    Rules (deterministic MVP):
    - ``lean_long``: useful if return > 0, harmful if return < 0.
    - ``lean_short``: useful if return < 0, harmful if return > 0.
    - ``reduce_exposure``: useful if return <= 0 or return is weakly positive (< +0.5%).
      Harmful when market is strongly positive (>= +1%).
    - ``avoid_risk``: useful if return <= -1%, or if 10d vol is elevated (>= 80th pct
      is handled upstream by providing a boolean-style proxy as numeric > 0.8).
      Harmful if return >= +1% and volatility is not elevated.
    - ``wait``: neutral by default, harmful when market makes a strong immediate move
      (abs return >= 1%).
    - ``watch``: neutral by default, harmful on strong moves (abs return >= 1.5%).
    """
    if pd.isna(forward_ret):
        return 0

    a = str(action)
    r = float(forward_ret)
    vol = float(vol_10d) if vol_10d is not None and not pd.isna(vol_10d) else np.nan

    if a == "lean_long":
        return 1 if r > 0 else (-1 if r < 0 else 0)

    if a == "lean_short":
        return 1 if r < 0 else (-1 if r > 0 else 0)

    if a == "reduce_exposure":
        if r <= 0.0 or r < 0.005:
            return 1
        if r >= 0.01:
            return -1
        return 0

    if a == "avoid_risk":
        vol_elevated = (not pd.isna(vol)) and (vol >= 0.8)
        if r <= -0.01 or vol_elevated:
            return 1
        if r >= 0.01 and not vol_elevated:
            return -1
        return 0

    if a == "wait":
        return -1 if abs(r) >= 0.01 else 0

    if a == "watch":
        return -1 if abs(r) >= 0.015 else 0

    return 0


def score_action_usefulness(
    df: pd.DataFrame,
    action_col: str = "action_posture",
    out_prefix: str = "action_useful",
) -> pd.DataFrame:
    """
    Score action usefulness at each available horizon.

    Requires the given ``action_col`` and forward return columns named ``fwd_ret_{h}d``.
    Creates ``{out_prefix}_{h}d`` columns with values in ``{-1, 0, 1}``.

    Default preserves Day 5 behavior: ``action_posture`` -> ``action_useful_{h}d``.
    """
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    if action_col not in out.columns:
        raise KeyError(f"Missing required column: {action_col}")

    vol_col = "spy_vol_10d" if "spy_vol_10d" in out.columns else None
    vol_vals: Iterable[float | None]
    vol_vals = out[vol_col] if vol_col else [None] * len(out)

    fwd_cols = [c for c in out.columns if c.startswith("fwd_ret_") and c.endswith("d")]
    for col in fwd_cols:
        horizon = col.replace("fwd_ret_", "").replace("d", "")
        out[f"{out_prefix}_{horizon}d"] = [
            _score_single_action(a, r, v)
            for a, r, v in zip(out[action_col], out[col], vol_vals)
        ]

    return out.loc[:, ~out.columns.duplicated()]


def evaluate_confidence_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize confidence calibration by fixed confidence bins.

    Uses average action usefulness across available horizons as the outcome proxy,
    then groups by confidence bins:
    [0.0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0].

    Returns columns:
    - confidence_bin
    - count
    - avg_usefulness
    - hit_rate (share with usefulness > 0)
    - non_neutral_rate (share with usefulness != 0)
    - monotonic_non_decreasing (global diagnostic repeated per row)

    The monotonicity diagnostic is true when ``avg_usefulness`` is non-decreasing
    across bins with data.
    """
    if "confidence_score" not in df.columns:
        raise KeyError("Missing required column: confidence_score")

    out = df.copy()
    useful_cols = [c for c in out.columns if c.startswith("action_useful_")]
    if not useful_cols:
        raise KeyError("No action usefulness columns found. Run score_action_usefulness first.")

    out["avg_action_usefulness"] = out[useful_cols].mean(axis=1)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    out["confidence_bin"] = pd.cut(
        out["confidence_score"].clip(0.0, 1.0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    summary = (
        out.groupby("confidence_bin", observed=False)
        .agg(
            count=("avg_action_usefulness", "size"),
            avg_usefulness=("avg_action_usefulness", "mean"),
            hit_rate=("avg_action_usefulness", lambda s: float((s > 0).mean())),
            non_neutral_rate=("avg_action_usefulness", lambda s: float((s != 0).mean())),
        )
        .reset_index()
    )

    non_empty = summary[summary["count"] > 0]["avg_usefulness"].to_numpy(dtype=float)
    monotonic = bool(np.all(np.diff(non_empty) >= -1e-12)) if len(non_empty) > 1 else True
    summary["monotonic_non_decreasing"] = monotonic

    return summary
