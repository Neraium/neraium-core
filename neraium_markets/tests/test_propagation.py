"""Day 8: regime propagation and influence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.clustering import build_asset_panel
from neraium.propagation import (
    build_regime_propagation_table,
    compute_asset_influence_scores,
    compute_sector_influence_scores,
)
from neraium.structural import build_structural_snapshot
from neraium.features import build_feature_table


def _setup():
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    assets = ["spy", "qqq", "xlf"]
    merged = pd.DataFrame({DATE_COLUMN: idx})
    for a in assets:
        merged[a.upper()] = 100 + np.cumsum(rng.normal(0, 0.5, n))
    feat = build_feature_table(merged)
    struct = build_structural_snapshot(feat)
    panel = build_asset_panel(struct, assets)
    return panel, assets


def test_propagation_table_not_empty():
    panel, assets = _setup()
    prop = build_regime_propagation_table(panel, assets)
    assert not prop.empty
    assert "propagation_count" in prop.columns


def test_asset_influence_scores_exist():
    panel, assets = _setup()
    prop = build_regime_propagation_table(panel, assets)
    inf = compute_asset_influence_scores(prop)
    assert not inf.empty
    assert "influence_score" in inf.columns


def test_sector_influence_scores_exist():
    panel, assets = _setup()
    prop = build_regime_propagation_table(panel, assets)
    sec_map = {"spy": "a", "qqq": "b", "xlf": "c"}
    s = compute_sector_influence_scores(prop, sec_map)
    assert not s.empty
    assert "sector" in s.columns
