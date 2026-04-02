from __future__ import annotations

import numpy as np
import pandas as pd

from neraium.clustering import cluster_assets, compute_asset_similarity_matrix, summarize_clusters


def _asset_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    assets = ["spy", "qqq", "xlf"]
    rows = []
    for a in assets:
        base = {"spy": 0.004, "qqq": 0.005, "xlf": -0.001}[a]
        for i, ts in enumerate(idx):
            ret = base + (0.001 if i % 3 == 0 else -0.0005)
            regime = "stable_trend" if ret > 0 else "risk_off_transition"
            action = "lean_long" if ret > 0 else "reduce_exposure"
            rows.append(
                {
                    "timestamp": ts,
                    "asset": a,
                    "ret_1d": ret,
                    "regime_label": regime,
                    "action_posture": action,
                    "adjusted_confidence_score": float(np.clip(0.6 + ret * 20, 0.0, 1.0)),
                    "action_useful_5d": 1.0 if ret > 0 else -1.0,
                    "structural_score": float(np.clip(0.5 + ret * 10, 0.0, 1.0)),
                }
            )
    return pd.DataFrame(rows)


def test_similarity_matrix_exists():
    df = _asset_df()
    assets = ["spy", "qqq", "xlf"]
    sim = compute_asset_similarity_matrix(df, assets)
    assert not sim.empty
    assert sim.shape == (len(assets), len(assets))
    assert list(sim.index) == assets
    assert list(sim.columns) == assets


def test_cluster_output_exists():
    sim = compute_asset_similarity_matrix(_asset_df(), ["spy", "qqq", "xlf"])
    clusters = cluster_assets(sim)
    assert {"asset", "cluster_id"}.issubset(clusters.columns)


def test_cluster_summary_not_empty():
    df = _asset_df()
    sim = compute_asset_similarity_matrix(df, ["spy", "qqq", "xlf"])
    clusters = cluster_assets(sim)
    summary = summarize_clusters(clusters, df)
    assert not summary.empty
    assert {"dominant_regime", "average_confidence", "dominant_action_posture"}.issubset(summary.columns)
