"""Tests for CSV loading."""

from __future__ import annotations

import pandas as pd

from neraium.data_loader import load_all_assets, load_asset_csv


def test_load_spy_has_required_columns():
    df = load_asset_csv("spy")
    cols = set(df.columns.str.lower())
    assert "timestamp" in cols
    assert "close" in cols
    assert "open" in cols
    assert "high" in cols
    assert "low" in cols
    assert "volume" in cols


def test_load_spy_sorted_ascending():
    df = load_asset_csv("spy")
    assert df["timestamp"].is_monotonic_increasing


def test_load_spy_at_least_30_rows():
    df = load_asset_csv("spy")
    assert len(df) >= 30


def test_load_all_assets_count_matches_config():
    import config

    data = load_all_assets()
    assert len(data) == len(config.ASSETS)
    assert set(data.keys()) == {a.lower() for a in config.ASSETS}


def test_timestamps_are_datetime():
    df = load_asset_csv("qqq")
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
