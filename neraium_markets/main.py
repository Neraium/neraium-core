#!/usr/bin/env python3
"""Neraium Markets Days 7–9: alignment, market state, and trajectory / path-aware warnings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ASSET_TO_SECTOR, DATE_COLUMN, DAY8_CLUSTER_ASSETS  # noqa: E402
from neraium.alignment import align_close_series  # noqa: E402
from neraium.alignment_filters import (  # noqa: E402
    apply_timeframe_alignment_filter,
    compare_alignment_filtered_vs_unfiltered,
)
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
    build_day7_alignment_report,
    build_day9_report,
    build_validation_report,
    save_day6_outputs,
    save_day7_outputs,
    save_day8_outputs,
    save_day9_outputs,
    save_validation_outputs,
)
from neraium.signals import generate_signals  # noqa: E402
from neraium.scenarios import label_scenario_paths, summarize_scenario_paths  # noqa: E402
from neraium.structural import build_structural_snapshot  # noqa: E402
from neraium.trajectories import compute_market_state_runs, compute_market_state_trajectory  # noqa: E402
from neraium.timeframe_alignment import (  # noqa: E402
    apply_timeframe_confidence_adjustment,
    build_timeframe_alignment_table,
    compute_action_agreement,
    compute_regime_agreement,
)
from neraium.transitions import (  # noqa: E402
    build_transition_matrix,
    compute_regime_runs,
    summarize_regime_persistence,
    summarize_transition_quality,
)
from neraium.validation import validate_all  # noqa: E402
from neraium.warnings import (  # noqa: E402
    adjust_market_action_for_path,
    compare_path_aware_vs_static_market_usefulness,
    compute_early_warning_flags,
    compute_path_persistence_score,
    compute_reversal_risk_score,
    generate_market_warning_level,
    generate_path_aware_market_explanation,
    path_aware_usefulness_breakdowns,
    refine_early_warnings_with_reversal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neraium Markets pipeline (Day 7–9 alignment, market state, trajectories)"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save Day 5/6/7/8/9 CSV/JSON artifacts under output/",
    )
    return parser.parse_args()


def _expand_timeframe_from_daily(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Deterministically expand daily close table into 1h or 15m synthetic bars."""
    if timeframe == "daily":
        return df.sort_values("timestamp", ascending=True).reset_index(drop=True)

    if timeframe == "1h":
        steps, freq = 7, "60min"
    elif timeframe == "15m":
        steps, freq = 26, "15min"
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    base = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    price_cols = [c for c in base.columns if c != "timestamp"]
    rows: list[dict[str, object]] = []

    for i in range(len(base)):
        ts = pd.Timestamp(base.loc[i, "timestamp"])
        intra_idx = pd.date_range(ts + pd.Timedelta(hours=9, minutes=30), periods=steps, freq=freq)

        nxt = base.iloc[i + 1] if i + 1 < len(base) else base.iloc[i]
        cur = base.iloc[i]

        for j, bar_ts in enumerate(intra_idx):
            alpha = float(j + 1) / float(steps)
            row: dict[str, object] = {"timestamp": bar_ts}
            for col in price_cols:
                start = float(cur[col])
                end = float(nxt[col])
                row[col] = start + alpha * (end - start)
            rows.append(row)

    return pd.DataFrame(rows).sort_values("timestamp", ascending=True).reset_index(drop=True)


def _run_pipeline_for_timeframe(merged_prices: pd.DataFrame, timeframe: str) -> dict[str, object]:
    price_df = _expand_timeframe_from_daily(merged_prices, timeframe)
    features = build_feature_table(price_df)
    structural = build_structural_snapshot(features)
    signals = generate_signals(structural)
    signals["timeframe"] = timeframe

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

    return {
        "signals": filtered,
        "summary": summary,
        "calibration": calibration,
        "baseline": baseline_comparison,
        "persistence": persistence_summary,
        "transition_matrix": transition_matrix,
        "transition_quality": transition_quality,
        "filtered_comparison": filtered_comparison,
        "day6_report": day6_report,
        "structural": structural,
        "evaluated": evaluated,
    }


def _build_asset_signal_table(merged_prices: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic per-asset state table needed for Day 8 cross-asset analysis."""
    price_cols = [c for c in merged_prices.columns if c != "timestamp"]
    col_map = {c.lower(): c for c in price_cols}
    selected = [a for a in ASSETS if a in col_map]
    if not selected:
        return pd.DataFrame(columns=["timestamp", "asset", "close"])

    long = (
        merged_prices.melt(
            id_vars=["timestamp"],
            value_vars=[col_map[a] for a in selected],
            var_name="asset",
            value_name="close",
        )
        .assign(asset=lambda d: d["asset"].astype(str).str.lower())
        .sort_values(["asset", "timestamp"], ascending=True)
        .reset_index(drop=True)
    )
    long["ret_1d"] = long.groupby("asset", observed=False)["close"].pct_change().fillna(0.0)
    long["ret_5d"] = long.groupby("asset", observed=False)["close"].pct_change(5).fillna(0.0)
    long["vol_10d"] = (
        long.groupby("asset", observed=False)["ret_1d"]
        .rolling(10, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    # Deterministic per-asset local regime proxy for cross-asset state analysis.
    conditions = [
        (long["ret_5d"] < -0.02) & (long["vol_10d"] > long["vol_10d"].quantile(0.7)),
        (long["vol_10d"] > long["vol_10d"].quantile(0.8)),
        (long["ret_5d"] > 0.015) & (long["vol_10d"] < long["vol_10d"].quantile(0.5)),
    ]
    choices = ["risk_off_transition", "high_volatility", "stable_trend"]
    long["regime_label"] = pd.Series(np.select(conditions, choices, default="mean_reversion"), dtype="object")

    long["action_posture"] = "watch"
    long.loc[long["regime_label"] == "stable_trend", "action_posture"] = "lean_long"
    long.loc[long["regime_label"].isin(["risk_off_transition", "high_volatility"]), "action_posture"] = "reduce_exposure"

    conf = (1.0 - (long["vol_10d"] / max(float(long["vol_10d"].quantile(0.95)), 1e-9))).clip(0.0, 1.0)
    long["adjusted_confidence_score"] = (0.55 * conf + 0.45 * (long["ret_5d"].abs().clip(upper=0.05) / 0.05)).clip(0.0, 1.0)

    # usefulness proxy consistent with prior days: direction agreement to forward return
    long["fwd_ret_5d"] = long.groupby("asset", observed=False)["ret_1d"].shift(-5).fillna(0.0)
    long["action_useful_5d"] = 0.0
    long.loc[(long["action_posture"] == "lean_long") & (long["fwd_ret_5d"] > 0), "action_useful_5d"] = 1.0
    long.loc[(long["action_posture"] == "lean_long") & (long["fwd_ret_5d"] < 0), "action_useful_5d"] = -1.0
    long.loc[(long["action_posture"] == "reduce_exposure") & (long["fwd_ret_5d"] < 0), "action_useful_5d"] = 1.0
    long.loc[(long["action_posture"] == "reduce_exposure") & (long["fwd_ret_5d"] > 0), "action_useful_5d"] = -1.0

    long["structural_score"] = (0.6 * (1.0 - long["vol_10d"].rank(pct=True)) + 0.4 * long["ret_5d"].rank(pct=True)).clip(0.0, 1.0)
    return long.sort_values(["timestamp", "asset"], ascending=True).reset_index(drop=True)


def _asset_to_sector_map() -> dict[str, str]:
    sector_map = {a: f"sector_{a}" for a in SECTOR_ASSETS}
    sector_map.update({"spy": "broad_equity", "qqq": "broad_equity", "iwm": "broad_equity"})
    sector_map.update({a: "macro_context" for a in CROSS_ASSET_CONTEXT})
    return sector_map


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

    daily = _run_pipeline_for_timeframe(merged, "daily")
    hourly = _run_pipeline_for_timeframe(merged, "1h")
    intraday = _run_pipeline_for_timeframe(merged, "15m")

    structural_i = intraday["structural"]
    evaluated_i = intraday["evaluated"]
    assets = list(dict.fromkeys(DAY8_CLUSTER_ASSETS))

    asset_panel = build_asset_panel(structural_i, assets)
    asset_panel = panel_attach_forwards(asset_panel, merged, assets)
    asset_panel = score_panel_action_usefulness(asset_panel)

    similarity = compute_asset_similarity_matrix(asset_panel, assets, structural_i)
    clusters = cluster_assets(similarity)
    cluster_summary = summarize_clusters(clusters, asset_panel)

    propagation = build_regime_propagation_table(asset_panel, assets)
    asset_influence = compute_asset_influence_scores(propagation)
    sector_influence = compute_sector_influence_scores(propagation, ASSET_TO_SECTOR)

    panel_agg = aggregate_panel_per_timestamp(asset_panel)
    wide_market = structural_i.merge(panel_agg, on=DATE_COLUMN, how="left")
    market_state = synthesize_market_state(wide_market, cluster_summary, asset_influence)
    market_state = generate_market_action_posture(market_state)
    market_state = generate_market_explanation(market_state)
    market_state = attach_spy_forward_returns(market_state, evaluated_i)

    market_state = compute_market_state_trajectory(market_state)
    market_state = compute_market_state_runs(market_state)
    market_state = label_scenario_paths(market_state)
    market_state = compute_path_persistence_score(market_state)
    market_state = compute_reversal_risk_score(market_state)
    market_state = compute_early_warning_flags(market_state)
    market_state = refine_early_warnings_with_reversal(market_state)
    market_state = generate_market_warning_level(market_state)
    market_state = adjust_market_action_for_path(market_state)
    market_state = generate_path_aware_market_explanation(market_state)

    scenario_path_summary = summarize_scenario_paths(market_state)
    path_comparison = compare_path_aware_vs_static_market_usefulness(market_state)
    by_scen, by_wl = path_aware_usefulness_breakdowns(market_state)
    day9_report = build_day9_report(
        market_state,
        scenario_path_summary,
        path_comparison,
        by_scen,
        by_wl,
    )

    market_vs_asset = compare_market_vs_asset_usefulness(asset_panel, market_state)

    alignment_df = build_timeframe_alignment_table(
        daily_df=daily["signals"],
        hourly_df=hourly["signals"],
        intraday_df=intraday["signals"],
    )
    alignment_df = compute_regime_agreement(alignment_df)
    alignment_df = compute_action_agreement(alignment_df)
    alignment_df = apply_timeframe_confidence_adjustment(alignment_df)
    alignment_df = apply_timeframe_alignment_filter(alignment_df)

    comparison = compare_alignment_filtered_vs_unfiltered(alignment_df)
    for _, row in comparison.iterrows():
        prefix = f"comparison_{row['version']}"
        for c in comparison.columns:
            if c != "version":
                alignment_df[f"{prefix}_{c}"] = row[c]
    alignment_df["comparison_version"] = "embedded"

    alignment_report = build_day7_alignment_report(alignment_df)

    summary = intraday["summary"]
    calibration = intraday["calibration"]
    baseline_comparison = intraday["baseline"]
    day6_report = intraday["day6_report"]
    filtered_comparison = intraday["filtered_comparison"]
    transition_matrix = intraday["transition_matrix"]

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

    improved_d6 = day6_report.get("filtering_improved_mean_usefulness", False)
    print(f"\nFiltering improved mean usefulness (1d/5d/10d avg): {improved_d6}")

    print("\n--- Day 7 multi-timeframe alignment ---")
    print("Total 15m aligned rows:", len(alignment_df))
    print("\nRegime alignment counts:")
    for k, v in alignment_report.get("regime_alignment_counts", {}).items():
        print(f"  {k}: {v}")

    print("\nAction alignment counts:")
    for k, v in alignment_report.get("action_alignment_counts", {}).items():
        print(f"  {k}: {v}")

    print(f"\nAverage adjusted confidence: {alignment_report['average_adjusted_confidence']:.4f}")
    print(f"Suppressed by alignment filter: {alignment_report['filter_suppression_count']}")

    print("\nAligned vs unaligned usefulness:")
    print(comparison.round(4).to_string(index=False))

    print("\nAsset clusters:")
    print(clustered_df.head(20).to_string(index=False))

    print("\nTop influence assets:")
    print(asset_influence_df.head(10).round(4).to_string(index=False))

    print("\nTop influence sectors:")
    print(sector_influence_df.head(10).round(4).to_string(index=False))

    print("\nMarket regime distribution:")
    print(market_state_df["market_regime_label"].value_counts().to_string())

    print("\nMarket action posture distribution:")
    print(market_state_df["market_action_posture"].value_counts().to_string())

    print("\nMarket vs Asset usefulness:")
    print(market_vs_asset_df.round(4).to_string(index=False))

    improved = False
    if len(comparison) >= 2:
        u = comparison.loc[comparison["version"] == "unaligned", ["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]].mean(axis=1)
        f = comparison.loc[comparison["version"] == "alignment_filtered", ["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]].mean(axis=1)
        improved = bool((f.iloc[0] if len(f) else 0.0) > (u.iloc[0] if len(u) else 0.0))
    print("\nMulti-timeframe alignment improved reliability:", improved)

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

    print("\n--- Day 9 trajectory & path intelligence ---")
    if "trajectory_direction" in market_state.columns:
        print("\nTrajectory direction counts:")
        print(market_state["trajectory_direction"].value_counts().to_string())
    if "market_warning_level" in market_state.columns:
        print("\nMarket warning level:")
        print(market_state["market_warning_level"].value_counts().to_string())
    print("\nPath vs static usefulness (mean 1d/5d/10d):")
    print(path_comparison.round(4).to_string(index=False))
    print("\nPath-adjusted improved mean usefulness:", day9_report.get("path_adjusted_improved_mean_usefulness", False))

    if args.save_output:
        out_dir = _ROOT / "output"
        base_paths = save_validation_outputs(
            signals_df=intraday["signals"],
            calibration_df=intraday["calibration"],
            baseline_df=intraday["baseline"],
            summary=intraday["summary"],
            output_dir=out_dir,
        )
        day6_paths = save_day6_outputs(
            persistence_summary=intraday["persistence"],
            transition_matrix=intraday["transition_matrix"],
            transition_quality=intraday["transition_quality"],
            filtered_comparison=intraday["filtered_comparison"],
            summary=intraday["day6_report"],
            output_dir=out_dir,
        )
        day7_paths = save_day7_outputs(
            alignment_df=alignment_df,
            comparison_df=comparison,
            summary=alignment_report,
            output_dir=out_dir,
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
        day9_paths = save_day9_outputs(
            market_state_day9_df=market_state,
            scenario_summary_df=scenario_path_summary,
            path_comparison_df=path_comparison,
            summary=day9_report,
            output_dir=_ROOT / "output",
        )

        day8_paths = {
            "asset_similarity_matrix": out_dir / "asset_similarity_matrix.csv",
            "asset_clusters": out_dir / "asset_clusters.csv",
            "cluster_summary": out_dir / "cluster_summary.csv",
            "regime_propagation": out_dir / "regime_propagation.csv",
            "asset_influence_scores": out_dir / "asset_influence_scores.csv",
            "sector_influence_scores": out_dir / "sector_influence_scores.csv",
            "market_state": out_dir / "market_state.csv",
            "market_vs_asset_comparison": out_dir / "market_vs_asset_comparison.csv",
        }
        similarity_df.to_csv(day8_paths["asset_similarity_matrix"])
        clustered_df.to_csv(day8_paths["asset_clusters"], index=False)
        cluster_summary_df.to_csv(day8_paths["cluster_summary"], index=False)
        propagation_df.to_csv(day8_paths["regime_propagation"], index=False)
        asset_influence_df.to_csv(day8_paths["asset_influence_scores"], index=False)
        sector_influence_df.to_csv(day8_paths["sector_influence_scores"], index=False)
        market_state_df.to_csv(day8_paths["market_state"], index=False)
        market_vs_asset_df.to_csv(day8_paths["market_vs_asset_comparison"], index=False)

        print("\nSaved outputs:")
        for dct in (base_paths, day6_paths, day7_paths, day8_paths, day9_paths):
            for key, path in dct.items():
                print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
