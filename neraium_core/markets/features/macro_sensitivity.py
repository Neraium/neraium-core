"""Macro sensitivity proxies."""

from __future__ import annotations

import pandas as pd


def compute_macro_sensitivity(returns: pd.DataFrame, target_assets: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    rates = returns[["US2Y", "US10Y"]].mean(axis=1)
    dollar = returns["DXY"]
    for asset in target_assets:
        out[f"{asset}_rates_beta"] = returns[asset].rolling(20, min_periods=5).corr(rates).fillna(0.0)
        out[f"{asset}_dxy_beta"] = returns[asset].rolling(20, min_periods=5).corr(dollar).fillna(0.0)
    return out
