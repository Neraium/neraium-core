"""Baseline and recent rolling window extraction."""

from __future__ import annotations

import pandas as pd


def split_baseline_recent(df: pd.DataFrame, baseline: int = 60, recent: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < baseline + recent:
        raise ValueError("Not enough samples for baseline + recent windows")
    return df.iloc[-(baseline + recent) : -recent], df.iloc[-recent:]
