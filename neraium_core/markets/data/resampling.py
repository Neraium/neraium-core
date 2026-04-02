"""Timeframe resampling helpers."""

from __future__ import annotations

import pandas as pd


def resample_prices(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample price matrix and forward-fill to keep alignment."""
    return df.resample(rule).last().ffill()
