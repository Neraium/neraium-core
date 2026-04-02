"""Breadth and sector participation proxies."""

from __future__ import annotations

import pandas as pd


def compute_breadth_features(returns: pd.DataFrame, core_assets: list[str], sector_assets: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    out["advancer_ratio"] = (returns[core_assets] > 0).sum(axis=1) / len(core_assets)
    out["sector_participation"] = (returns[sector_assets] > 0).sum(axis=1) / len(sector_assets)
    out["breadth_momentum"] = out["advancer_ratio"].diff().fillna(0.0)
    return out
