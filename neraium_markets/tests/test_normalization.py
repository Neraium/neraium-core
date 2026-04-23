"""Day 13: normalization."""

from __future__ import annotations

import pandas as pd

from neraium.normalization import CANONICAL_COLUMNS, normalize_market_dataframe, normalize_many_assets


def test_normalization_canonical_columns() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "Open": [10, 11],
            "High": [11, 12],
            "Low": [9, 10],
            "Close": [10.5, 11.5],
            "Vol": [100, 200],
        }
    )
    out = normalize_market_dataframe(raw, "test")
    assert list(out.columns) == list(CANONICAL_COLUMNS)


def test_normalization_sorts_timestamp() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-03", "2024-01-01", "2024-01-02"],
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [1, 1, 1],
        }
    )
    out = normalize_market_dataframe(raw, "test")
    ts = pd.to_datetime(out["timestamp"])
    assert ts.is_monotonic_increasing


def test_normalize_many_assets() -> None:
    a = pd.DataFrame(
        {
            "ts": ["2024-01-01"],
            "o": [1],
            "h": [1],
            "l": [1],
            "c": [1],
            "v": [1],
        }
    )
    m = normalize_many_assets({"spy": a}, "x")
    assert "spy" in m
    assert "timestamp" in m["spy"].columns
