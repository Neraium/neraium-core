#!/usr/bin/env python3
"""Neraium Markets Day 8: cross-asset clustering, propagation, and market-state synthesis."""

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

from config import ASSETS, CROSS_ASSET_CONTEXT, SECTOR_ASSETS
from neraium.alignment import align_close_series  # noqa: E402
from neraium.alignment_filters import (  # noqa: E402
    apply_timeframe_alignment_filter,
    compare_alignment_filtered_vs_unfiltered,
)
from neraium.baselines import compare_to_baselines  # noqa: E402
from neraium.clustering import (  # noqa: E402
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
    compare_market_vs_asset_usefulness,
    generate_market_action_posture,
    generate_market_explanation,
    synthesize_market_state,
)
from neraium.propagation import (  # noqa: E402
    build_regime_propagation_table,
    compute_asset_influence_scores,
    compute_sector_influence_scores,
)
from neraium.reporting import (  # noqa: E402
    build_day6_reliability_report,
    build_day7_alignment_report,
    build_validation_report,
    save_day6_outputs,
    save_day7_outputs,
    save_validation_outputs,
)
from neraium.signals import generate_signals  # noqa: E402
from neraium.structural import build_structural_snapshot  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neraium Markets Day 8 cross-asset/market-state pipeline"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save Day 5/6/7/8 CSV/JSON artifacts under output/",
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

    # Day 8 cross-asset intelligence layer
    asset_signal_df = _build_asset_signal_table(merged)
    similarity_df = compute_asset_similarity_matrix(asset_signal_df, ASSETS)
    clustered_df = cluster_assets(similarity_df)
    cluster_summary_df = summarize_clusters(clustered_df, asset_signal_df)

    propagation_df = build_regime_propagation_table(asset_signal_df, ASSETS)
    asset_influence_df = compute_asset_influence_scores(propagation_df)
    sector_influence_df = compute_sector_influence_scores(propagation_df, _asset_to_sector_map())

    market_state_df = synthesize_market_state(asset_signal_df, cluster_summary_df, asset_influence_df)
    market_state_df = generate_market_action_posture(market_state_df)
    market_state_df = generate_market_explanation(market_state_df)
    market_state_df["market_usefulness_proxy"] = float(asset_signal_df["action_useful_5d"].mean())

    market_vs_asset_df = compare_market_vs_asset_usefulness(asset_signal_df, market_state_df)

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
        for dct in (base_paths, day6_paths, day7_paths, day8_paths):
            for key, path in dct.items():
                print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
