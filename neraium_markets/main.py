#!/usr/bin/env python3
"""Neraium Markets Day 8: validation + reliability + cross-asset market state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ASSET_TO_SECTOR, DATE_COLUMN, DAY8_CLUSTER_ASSETS  # noqa: E402
from neraium.alignment import align_close_series  # noqa: E402
from neraium.baselines import compare_to_baselines  # noqa: E402
from neraium.clustering import (  # noqa: E402
    build_asset_panel,
    cluster_assets,
    compute_asset_similarity_matrix,
    summarize_clusters,
)
from neraium.data_loader import load_all_assets  # noqa: E402
from neraium.diagnostics import compute_signal_stability  # noqa: E402
from neraium.evaluation import (  # noqa: E402
    compute_forward_returns,
    evaluate_confidence_calibration,
    score_action_usefulness,
)
from neraium.features import build_feature_table  # noqa: E402
from neraium.filtering import apply_signal_filters, compare_filtered_vs_unfiltered, identify_false_positive_patterns  # noqa: E402
from neraium.market_state import (  # noqa: E402
    aggregate_panel_per_timestamp,
    attach_spy_forward_returns,
    compare_market_vs_asset_usefulness,
    generate_market_action_posture,
    generate_market_explanation,
    score_panel_action_usefulness,
    synthesize_market_state,
    attach_asset_forward_returns as panel_attach_forwards,
)
from neraium.propagation import (  # noqa: E402
    build_regime_propagation_table,
    compute_asset_influence_scores,
    compute_sector_influence_scores,
)
from neraium.reporting import (  # noqa: E402
    build_day6_reliability_report,
    build_validation_report,
    save_day6_outputs,
    save_day8_outputs,
    save_validation_outputs,
)
from neraium.signals import generate_signals  # noqa: E402
from neraium.structural import build_structural_snapshot  # noqa: E402
from neraium.transitions import (  # noqa: E402
    build_transition_matrix,
    compute_regime_runs,
    summarize_regime_persistence,
    summarize_transition_quality,
)
from neraium.validation import validate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neraium Markets Day 8 market-state pipeline"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save CSV/JSON artifacts under output/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data = load_all_assets()
    errors = validate_all(data)
    if errors:
        print("Validation failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    merged = align_close_series(data)
    features = build_feature_table(merged)
    structural = build_structural_snapshot(features)
    signals = generate_signals(structural)

    evaluated = compute_forward_returns(signals, price_col="spy", horizons=[1, 5, 10])
    evaluated = score_action_usefulness(evaluated)
    calibration = evaluate_confidence_calibration(evaluated)
    baseline_comparison = compare_to_baselines(evaluated)
    summary = build_validation_report(evaluated)

    with_runs = compute_regime_runs(evaluated)
    persistence_summary = summarize_regime_persistence(with_runs)
    transition_matrix = build_transition_matrix(with_runs)
    stable = compute_signal_stability(with_runs)
    transition_quality = summarize_transition_quality(stable)
    flagged = identify_false_positive_patterns(stable)
    filtered = apply_signal_filters(flagged)
    filtered = score_action_usefulness(
        filtered,
        action_col="filtered_action_posture",
        out_prefix="filtered_useful",
    )
    filtered_comparison = compare_filtered_vs_unfiltered(filtered)
    day6_report = build_day6_reliability_report(
        filtered,
        persistence_summary,
        transition_matrix,
        transition_quality,
        filtered_comparison,
    )

    # --- Day 8: cross-asset clustering, propagation, market state ---
    assets = list(dict.fromkeys(DAY8_CLUSTER_ASSETS))

    asset_panel = build_asset_panel(structural, assets)
    asset_panel = panel_attach_forwards(asset_panel, merged, assets)
    asset_panel = score_panel_action_usefulness(asset_panel)

    similarity = compute_asset_similarity_matrix(asset_panel, assets, structural)
    clusters = cluster_assets(similarity)
    cluster_summary = summarize_clusters(clusters, asset_panel)

    propagation = build_regime_propagation_table(asset_panel, assets)
    asset_influence = compute_asset_influence_scores(propagation)
    sector_influence = compute_sector_influence_scores(propagation, ASSET_TO_SECTOR)

    panel_agg = aggregate_panel_per_timestamp(asset_panel)
    wide_market = structural.merge(panel_agg, on=DATE_COLUMN, how="left")
    market_state = synthesize_market_state(wide_market, cluster_summary, asset_influence)
    market_state = generate_market_action_posture(market_state)
    market_state = generate_market_explanation(market_state)
    market_state = attach_spy_forward_returns(market_state, evaluated)
    market_vs_asset = compare_market_vs_asset_usefulness(asset_panel, market_state)

    print("Total signals:", summary["total_signals"])
    print("\nCounts by regime:")
    for k, v in summary["regime_counts"].items():
        print(f"  {k}: {v}")

    print("\nCounts by action posture:")
    for k, v in summary["action_counts"].items():
        print(f"  {k}: {v}")

    print("\nAverage usefulness by horizon (unfiltered):")
    for k, v in summary["usefulness_summary"].items():
        print(f"  {k}: {v:.4f}")

    monotonic = bool(calibration["monotonic_non_decreasing"].iloc[0]) if not calibration.empty else False
    print("\nHigher confidence improved usefulness (monotonic bin check):", monotonic)

    print("\nBaseline comparison summary (avg_usefulness):")
    pivot = baseline_comparison.pivot(index="model", columns="horizon", values="avg_usefulness")
    print(pivot.round(4).to_string())

    print("\n--- Day 6 reliability ---")
    arl = day6_report.get("average_regime_run_length", float("nan"))
    print(f"Average regime run length: {arl:.4f}")

    if not transition_matrix.empty:
        top_t = transition_matrix.sort_values("transition_count", ascending=False).head(5)
        print("\nMost common regime transitions (top 5):")
        for _, r in top_t.iterrows():
            print(
                f"  {r['from_regime']} -> {r['to_regime']}: "
                f"count={int(r['transition_count'])} "
                f"p={float(r['p_to_given_from']):.3f}"
            )

    lfp = day6_report.get("false_positive_flag_counts", {}).get("likely_false_positive_flag", 0)
    print(f"\nCount of likely false positives (flag): {lfp}")

    print("\nFiltered vs unfiltered usefulness:")
    print(filtered_comparison.round(4).to_string(index=False))

    improved = day6_report.get("filtering_improved_mean_usefulness", False)
    print(f"\nFiltering improved mean usefulness (1d/5d/10d avg): {improved}")

    print("\n--- Day 8 market structure ---")
    print("\nAsset clusters (asset -> cluster_id):")
    print(clusters.to_string(index=False))

    print("\nTop influence assets:")
    if not asset_influence.empty:
        print(asset_influence.head(8).to_string(index=False))
    else:
        print("  (empty)")

    print("\nTop influence sectors:")
    if not sector_influence.empty:
        print(sector_influence.head(8).to_string(index=False))
    else:
        print("  (empty)")

    if "market_regime_label" in market_state.columns:
        print("\nMarket regime distribution:")
        print(market_state["market_regime_label"].value_counts().to_string())

    if "market_action_posture" in market_state.columns:
        print("\nMarket action posture distribution:")
        print(market_state["market_action_posture"].value_counts().to_string())

    print("\nMarket vs asset usefulness comparison:")
    print(market_vs_asset.round(4).to_string(index=False))

    if args.save_output:
        paths = save_validation_outputs(
            signals_df=filtered,
            calibration_df=calibration,
            baseline_df=baseline_comparison,
            summary=summary,
            output_dir=_ROOT / "output",
        )
        day6_paths = save_day6_outputs(
            persistence_summary=persistence_summary,
            transition_matrix=transition_matrix,
            transition_quality=transition_quality,
            filtered_comparison=filtered_comparison,
            summary=day6_report,
            output_dir=_ROOT / "output",
        )
        day8_paths = save_day8_outputs(
            similarity_df=similarity,
            clusters_df=clusters,
            cluster_summary_df=cluster_summary,
            propagation_df=propagation,
            asset_influence_df=asset_influence,
            sector_influence_df=sector_influence,
            market_state_df=market_state,
            market_vs_asset_df=market_vs_asset,
            output_dir=_ROOT / "output",
        )
        print("\nSaved outputs:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
        for key, path in day6_paths.items():
            print(f"  {key}: {path}")
        for key, path in day8_paths.items():
            print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
