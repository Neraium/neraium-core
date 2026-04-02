"""Day 8: market-wide state and comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATE_COLUMN
from neraium.clustering import build_asset_panel, cluster_assets, compute_asset_similarity_matrix, summarize_clusters
from neraium.market_state import (
    aggregate_panel_per_timestamp,
    attach_spy_forward_returns,
    compare_market_vs_asset_usefulness,
    generate_market_action_posture,
    generate_market_explanation,
    score_panel_action_usefulness,
    synthesize_market_state,
    attach_asset_forward_returns as panel_attach_forwards,
)
from neraium.propagation import build_regime_propagation_table, compute_asset_influence_scores
from neraium.evaluation import compute_forward_returns, score_action_usefulness
from neraium.features import build_feature_table
from neraium.structural import build_structural_snapshot


def _full():
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(2)
    assets = ["spy", "qqq", "xlf"]
    merged = pd.DataFrame({DATE_COLUMN: idx})
    for a in assets:
        merged[a.upper()] = 100 + np.cumsum(rng.normal(0, 0.5, n))
    feat = build_feature_table(merged)
    struct = build_structural_snapshot(feat)
    panel = build_asset_panel(struct, assets)
    panel = panel_attach_forwards(panel, merged, assets)
    panel = score_panel_action_usefulness(panel)
    sim = compute_asset_similarity_matrix(panel, assets, struct)
    clusters = cluster_assets(sim)
    cluster_summary = summarize_clusters(clusters, panel)
    propagation = build_regime_propagation_table(panel, assets)
    asset_inf = compute_asset_influence_scores(propagation)
    panel_agg = aggregate_panel_per_timestamp(panel)
    wide = struct.merge(panel_agg, on=DATE_COLUMN, how="left")
    ms = synthesize_market_state(wide, cluster_summary, asset_inf)
    ms = generate_market_action_posture(ms)
    ms = generate_market_explanation(ms)
    spy_eval = compute_forward_returns(
        struct.assign(action_posture="watch", confidence_score=0.5),
        price_col="spy",
        horizons=[1, 5, 10],
    )
    spy_eval = score_action_usefulness(spy_eval)
    ms = attach_spy_forward_returns(ms, spy_eval)
    return panel, ms


def test_market_state_columns_exist():
    panel, ms = _full()
    for c in ("market_regime_label", "market_confidence_score", "market_instability_score", "market_coherence_score"):
        assert c in ms.columns


def test_market_action_exists():
    _, ms = _full()
    assert "market_action_posture" in ms.columns


def test_market_explanation_exists():
    _, ms = _full()
    assert "market_explanation" in ms.columns


def test_market_vs_asset_comparison_not_empty():
    panel, ms = _full()
    cmp_df = compare_market_vs_asset_usefulness(panel, ms)
    assert not cmp_df.empty
    assert len(cmp_df) >= 2


def test_sorted_output_preserved():
    _, ms = _full()
    assert ms[DATE_COLUMN].is_monotonic_increasing


def test_no_duplicate_columns():
    _, ms = _full()
    assert ms.columns.is_unique
