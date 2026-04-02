"""Cross-asset context features."""

from __future__ import annotations

import pandas as pd


def compute_cross_asset_features(returns: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    out["equity_vs_dxy"] = returns[["SPY", "QQQ", "IWM"]].mean(axis=1) - returns["DXY"]
    out["equity_vs_rates"] = returns[["SPY", "QQQ", "IWM"]].mean(axis=1) - returns[["US2Y", "US10Y"]].mean(axis=1)
    out["commodity_inflation_pressure"] = returns[["CRUDE", "GOLD"]].mean(axis=1)
    out["vix_change"] = returns["VIX"]
    return out.fillna(0.0)
