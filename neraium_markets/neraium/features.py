from __future__ import annotations

import numpy as np
import pandas as pd

from config import CORE_EQUITY_ASSETS, CROSS_ASSET_CONTEXT, DATE_COLUMN, SECTOR_ASSETS


def compute_returns(df: pd.DataFrame, assets: list[str], windows: list[int] = [1, 5]) -> pd.DataFrame:
    """Add simple percentage return features for each asset and window."""
    out = df.copy()
    for asset in assets:
        if asset not in out.columns:
            continue
        for window in windows:
            out[f"{asset}_ret_{window}d"] = out[asset].pct_change(periods=window)
    return out


def compute_rolling_volatility(
    df: pd.DataFrame,
    assets: list[str],
    windows: list[int] = [10, 20],
) -> pd.DataFrame:
    """Add rolling volatility features based on 1-day returns."""
    out = df.copy()
    for asset in assets:
        ret_col = f"{asset}_ret_1d"
        if ret_col not in out.columns:
            continue
        for window in windows:
            out[f"{asset}_vol_{window}d"] = out[ret_col].rolling(window=window).std()
    return out


def compute_breadth_features(df: pd.DataFrame, sector_assets: list[str]) -> pd.DataFrame:
    """Add market breadth feature: % of sector ETFs above their 20-day moving average."""
    out = df.copy()

    above_cols: list[pd.Series] = []
    for asset in sector_assets:
        if asset not in out.columns:
            continue
        ma20 = out[asset].rolling(window=20).mean()
        above_cols.append((out[asset] > ma20).astype(float))

    if above_cols:
        stacked = pd.concat(above_cols, axis=1)
        out["breadth_pct_above_20dma"] = stacked.mean(axis=1) * 100.0
    else:
        out["breadth_pct_above_20dma"] = np.nan

    return out


def compute_sector_participation(df: pd.DataFrame, sector_assets: list[str]) -> pd.DataFrame:
    """Add sector participation features based on 1-day sector returns."""
    out = df.copy()

    ret_cols: list[str] = []
    for asset in sector_assets:
        col = f"{asset}_ret_1d"
        if col in out.columns:
            ret_cols.append(col)

    if not ret_cols:
        out["sector_dispersion_1d"] = np.nan
        out["sector_concentration_top2"] = np.nan
        return out

    sector_rets = out[ret_cols]
    out["sector_dispersion_1d"] = sector_rets.std(axis=1)

    abs_rets = sector_rets.abs()
    top2_sum = abs_rets.apply(lambda row: row.nlargest(2).sum(), axis=1)
    total = abs_rets.sum(axis=1)
    out["sector_concentration_top2"] = np.where(total > 0, top2_sum / total, np.nan)

    return out


def compute_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-asset context features and deterministic risk-off proxy."""
    out = df.copy()

    out = compute_returns(out, assets=CROSS_ASSET_CONTEXT, windows=[1, 5])

    if {"us10y", "us2y"}.issubset(out.columns):
        out["rates_2s10s"] = out["us10y"] - out["us2y"]
    else:
        out["rates_2s10s"] = np.nan

    terms = []
    for col, sign in [
        ("vix_ret_1d", 1.0),
        ("dxy_ret_1d", 1.0),
        ("us10y_ret_1d", -1.0),
        ("oil_ret_1d", -1.0),
    ]:
        if col in out.columns:
            terms.append(sign * out[col].fillna(0.0))

    out["risk_off_proxy"] = sum(terms) if terms else np.nan
    return out


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the Day 2 engineered feature table from aligned close data."""
    out = df.copy().sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)

    out = compute_returns(out, assets=CORE_EQUITY_ASSETS + SECTOR_ASSETS, windows=[1, 5])
    out = compute_rolling_volatility(out, assets=CORE_EQUITY_ASSETS, windows=[10, 20])
    out = compute_breadth_features(out, sector_assets=SECTOR_ASSETS)
    out = compute_sector_participation(out, sector_assets=SECTOR_ASSETS)
    out = compute_cross_asset_features(out)

    out = out.loc[:, ~out.columns.duplicated()]
    return out
