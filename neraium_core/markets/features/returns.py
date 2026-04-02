"""Return features."""

from __future__ import annotations

import pandas as pd


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return prices.pct_change(periods=periods).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
