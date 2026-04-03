"""Cross-asset context features."""

from __future__ import annotations

import pandas as pd


def compute_cross_asset_features(returns: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    equities = returns.reindex(columns=["SPY", "QQQ", "IWM"], fill_value=0.0).mean(axis=1)
    dxy = returns.get("DXY", pd.Series(0.0, index=returns.index))
    rates = returns.reindex(columns=["US2Y", "US10Y"], fill_value=0.0).mean(axis=1)
    out["equity_vs_dxy"] = equities - dxy
    out["equity_vs_rates"] = equities - rates
    out["commodity_inflation_pressure"] = returns.reindex(columns=["CRUDE", "GOLD"], fill_value=0.0).mean(axis=1)
    out["vix_change"] = returns.get("VIX", pd.Series(0.0, index=returns.index))
    return out.fillna(0.0)
