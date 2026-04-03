"""Macro sensitivity proxies."""

from __future__ import annotations

import pandas as pd


def compute_macro_sensitivity(returns: pd.DataFrame, target_assets: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    rates = returns.reindex(columns=["US2Y", "US10Y"], fill_value=0.0).mean(axis=1)
    dollar = returns.get("DXY", pd.Series(0.0, index=returns.index))
    for asset in target_assets:
        series = returns.get(asset, pd.Series(0.0, index=returns.index))
        out[f"{asset}_rates_beta"] = series.rolling(20, min_periods=5).corr(rates).fillna(0.0)
        out[f"{asset}_dxy_beta"] = series.rolling(20, min_periods=5).corr(dollar).fillna(0.0)
    return out
