"""Tests for timestamp alignment."""

from __future__ import annotations

import config
from neraium.alignment import align_close_series
from neraium.data_loader import load_all_assets


def test_alignment_column_count():
    data = load_all_assets()
    merged = align_close_series(data)
    # timestamp + one column per asset
    assert merged.shape[1] == 1 + len(config.ASSETS)


def test_alignment_includes_timestamp_and_all_symbols():
    data = load_all_assets()
    merged = align_close_series(data)
    assert config.DATE_COLUMN in merged.columns
    for asset in config.ASSETS:
        assert asset.upper() in merged.columns


def test_merged_sort_order():
    data = load_all_assets()
    merged = align_close_series(data)
    assert merged[config.DATE_COLUMN].is_monotonic_increasing


def test_outer_join_preserves_union_of_dates():
    import pandas as pd

    from neraium.data_loader import load_asset_csv

    a = load_asset_csv("spy").iloc[:10].copy()
    b = load_asset_csv("qqq").iloc[:12].copy()
    merged = align_close_series({"spy": a, "qqq": b})
    assert merged[config.DATE_COLUMN].is_monotonic_increasing
    exp = pd.concat([a["timestamp"], b["timestamp"]]).drop_duplicates().sort_values().reset_index(drop=True)
    got = merged[config.DATE_COLUMN].reset_index(drop=True)
    pd.testing.assert_series_equal(got, exp, check_names=False)
