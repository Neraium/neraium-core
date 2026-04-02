#!/usr/bin/env python3
"""Neraium Markets Day 7: multi-timeframe alignment + confidence-aware filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.alignment import align_close_series  # noqa: E402
from neraium.alignment_filters import (  # noqa: E402
    apply_timeframe_alignment_filter,
    compare_alignment_filtered_vs_unfiltered,
)
from neraium.baselines import compare_to_baselines  # noqa: E402
from neraium.data_loader import load_all_assets  # noqa: E402
from neraium.diagnostics import compute_signal_stability  # noqa: E402
from neraium.evaluation import (  # noqa: E402
    compute_forward_returns,
    evaluate_confidence_calibration,
    score_action_usefulness,
)
from neraium.features import build_feature_table  # noqa: E402
from neraium.filtering import apply_signal_filters, compare_filtered_vs_unfiltered, identify_false_positive_patterns  # noqa: E402
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
        description="Run Neraium Markets Day 7 multi-timeframe pipeline"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save Day 5/6/7 CSV/JSON artifacts under output/",
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

    improved = False
    if len(comparison) >= 2:
        u = comparison.loc[comparison["version"] == "unaligned", ["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]].mean(axis=1)
        f = comparison.loc[comparison["version"] == "alignment_filtered", ["avg_usefulness_1d", "avg_usefulness_5d", "avg_usefulness_10d"]].mean(axis=1)
        improved = bool((f.iloc[0] if len(f) else 0.0) > (u.iloc[0] if len(u) else 0.0))
    print("\nMulti-timeframe alignment improved reliability:", improved)

    if args.save_output:
        base_paths = save_validation_outputs(
            signals_df=intraday["signals"],
            calibration_df=intraday["calibration"],
            baseline_df=intraday["baseline"],
            summary=intraday["summary"],
            output_dir=_ROOT / "output",
        )
        day6_paths = save_day6_outputs(
            persistence_summary=intraday["persistence"],
            transition_matrix=intraday["transition_matrix"],
            transition_quality=intraday["transition_quality"],
            filtered_comparison=intraday["filtered_comparison"],
            summary=intraday["day6_report"],
            output_dir=_ROOT / "output",
        )
        day7_paths = save_day7_outputs(
            alignment_df=alignment_df,
            comparison_df=comparison,
            summary=alignment_report,
            output_dir=_ROOT / "output",
        )

        print("\nSaved outputs:")
        for dct in (base_paths, day6_paths, day7_paths):
            for key, path in dct.items():
                print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
