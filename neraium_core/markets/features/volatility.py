"""Volatility features."""

from __future__ import annotations

import pandas as pd


def realized_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return returns.rolling(window=window, min_periods=5).std().fillna(0.0)
