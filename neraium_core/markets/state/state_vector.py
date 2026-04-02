"""Build multidimensional market state vectors."""

from __future__ import annotations

import pandas as pd

from neraium_core.markets.features.breadth import compute_breadth_features
from neraium_core.markets.features.cross_asset import compute_cross_asset_features
from neraium_core.markets.features.macro_sensitivity import compute_macro_sensitivity
from neraium_core.markets.features.returns import compute_returns
from neraium_core.markets.features.volatility import realized_volatility

CORE = ["SPY", "QQQ", "IWM"]
SECTORS = ["XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU"]


def build_state_vector(prices: pd.DataFrame) -> pd.DataFrame:
    rets = compute_returns(prices)
    vol = realized_volatility(rets)[CORE].add_prefix("vol_")
    breadth = compute_breadth_features(rets, CORE, SECTORS)
    cross = compute_cross_asset_features(rets)
    macro = compute_macro_sensitivity(rets, CORE)
    state = pd.concat(
        [
            rets[CORE].add_prefix("ret_"),
            vol,
            breadth,
            cross,
            macro,
        ],
        axis=1,
    )
    return state.fillna(0.0)
