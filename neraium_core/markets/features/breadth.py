"""Breadth and sector participation proxies."""

from __future__ import annotations

import pandas as pd


def compute_breadth_features(returns: pd.DataFrame, core_assets: list[str], sector_assets: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    core = returns.reindex(columns=core_assets, fill_value=0.0)
    sectors = returns.reindex(columns=sector_assets, fill_value=0.0)
    out["advancer_ratio"] = (core > 0).sum(axis=1) / len(core_assets)
    out["sector_participation"] = (sectors > 0).sum(axis=1) / len(sector_assets)
    out["breadth_momentum"] = out["advancer_ratio"].diff().fillna(0.0)
    return out
