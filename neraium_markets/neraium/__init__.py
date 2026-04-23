"""Neraium Markets: read-only loading, features, structural snapshot, and signals."""

from neraium.alignment import align_close_series
from neraium.clustering import cluster_assets, compute_asset_similarity_matrix, summarize_clusters
from neraium.market_state import (
    compare_market_vs_asset_usefulness,
    generate_market_action_posture,
    generate_market_explanation,
    synthesize_market_state,
)
from neraium.propagation import (
    build_regime_propagation_table,
    compute_asset_influence_scores,
    compute_sector_influence_scores,
)
from neraium.data_loader import load_all_assets, load_asset_csv
from neraium.regime import (
    apply_interpretive_gate,
    classify_regime,
    compute_confidence_score,
    generate_action_posture,
)
from neraium.signals import generate_signals, run_full_signal_pipeline, save_signals_csv
from neraium.structural import build_structural_snapshot
from neraium.validation import validate_all, validate_dataframe

__all__ = [
    "load_asset_csv",
    "load_all_assets",
    "validate_dataframe",
    "validate_all",
    "align_close_series",
    "build_structural_snapshot",
    "classify_regime",
    "compute_confidence_score",
    "apply_interpretive_gate",
    "generate_action_posture",
    "generate_signals",
    "run_full_signal_pipeline",
    "save_signals_csv",
    "compute_asset_similarity_matrix",
    "cluster_assets",
    "summarize_clusters",
    "build_regime_propagation_table",
    "compute_asset_influence_scores",
    "compute_sector_influence_scores",
    "synthesize_market_state",
    "generate_market_action_posture",
    "generate_market_explanation",
    "compare_market_vs_asset_usefulness",
]
