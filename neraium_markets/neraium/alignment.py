"""Align close prices from multiple assets onto a common timestamp index."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


def _symbol_column(asset: str) -> str:
    return asset.strip().upper()


def align_close_series(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Outer-join all assets on timestamp.

    Each asset contributes its close series renamed to the ticker symbol (uppercase).
    Output columns: timestamp, then one column per asset in sorted symbol order.
    Rows sorted ascending by timestamp.
    """
    if not data:
        return pd.DataFrame(columns=[config.DATE_COLUMN])

    pieces: list[pd.DataFrame] = []
    for asset in sorted(data.keys()):
        df = data[asset]
        sym = _symbol_column(asset)
        sub = df[[config.DATE_COLUMN, config.PRICE_COLUMN]].copy()
        sub = sub.rename(columns={config.PRICE_COLUMN: sym})
        pieces.append(sub)

    merged = pieces[0]
    for nxt in pieces[1:]:
        merged = merged.merge(nxt, on=config.DATE_COLUMN, how="outer")

    merged = merged.sort_values(config.DATE_COLUMN, ascending=True).reset_index(drop=True)
    return merged
