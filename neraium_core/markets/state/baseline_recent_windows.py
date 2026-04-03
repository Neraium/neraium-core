"""Baseline and recent rolling window extraction."""

from __future__ import annotations

import pandas as pd


def split_baseline_recent(
    df: pd.DataFrame,
    baseline: int = 60,
    recent: int = 20,
    min_baseline: int = 20,
    min_recent: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < min_baseline + min_recent:
        raise ValueError("Not enough samples for baseline + recent windows")
    if len(df) < baseline + recent:
        recent = max(min_recent, min(recent, len(df) // 4))
        baseline = len(df) - recent
        if baseline < min_baseline:
            baseline = min_baseline
            recent = len(df) - baseline
    return df.iloc[-(baseline + recent) : -recent], df.iloc[-recent:]
