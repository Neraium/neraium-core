"""Day 8: asset similarity and clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.clustering import (
    build_asset_panel,
    cluster_assets,
    compute_asset_similarity_matrix,
    summarize_clusters,
)
from neraium.market_state import score_panel_action_usefulness, attach_asset_forward_returns
from neraium.structural import build_structural_snapshot
from neraium.features import build_feature_table


def _mini_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Synthetic aligned prices + features + structural + assets."""
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    assets = ["spy", "qqq", "xlf"]
    merged = pd.DataFrame({DATE_COLUMN: idx})
    for a in assets:
        merged[a.upper()] = 100 + np.cumsum(rng.normal(0, 0.5, n))
    feat = build_feature_table(merged)
    struct = build_structural_snapshot(feat)
    return merged, feat, struct, assets


def test_similarity_matrix_exists():
    merged, _feat, struct, assets = _mini_pipeline()
    panel = build_asset_panel(struct, assets)
    panel = attach_asset_forward_returns(panel, merged, assets)
    panel = score_panel_action_usefulness(panel)
    sim = compute_asset_similarity_matrix(panel, assets, struct)
    assert sim.shape[0] == sim.shape[1]
    assert list(sim.index) == list(sim.columns)
    assert set(sim.index) == set(assets)


def test_cluster_output_exists():
    merged, _feat, struct, assets = _mini_pipeline()
    panel = build_asset_panel(struct, assets)
    sim = compute_asset_similarity_matrix(panel, assets, struct)
    cl = cluster_assets(sim)
    assert "asset" in cl.columns
    assert "cluster_id" in cl.columns


def test_cluster_summary_not_empty():
    merged, _feat, struct, assets = _mini_pipeline()
    panel = build_asset_panel(struct, assets)
    panel = attach_asset_forward_returns(panel, merged, assets)
    panel = score_panel_action_usefulness(panel)
    sim = compute_asset_similarity_matrix(panel, assets, struct)
    cl = cluster_assets(sim)
    summ = summarize_clusters(cl, panel)
    assert not summ.empty
