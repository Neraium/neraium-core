#!/usr/bin/env python3
"""Neraium Markets Day 5: pipeline + validation and baseline comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.alignment import align_close_series  # noqa: E402
from neraium.baselines import compare_to_baselines  # noqa: E402
from neraium.data_loader import load_all_assets  # noqa: E402
from neraium.evaluation import (  # noqa: E402
    compute_forward_returns,
    evaluate_confidence_calibration,
    score_action_usefulness,
)
from neraium.features import build_feature_table  # noqa: E402
from neraium.reporting import build_validation_report, save_validation_outputs  # noqa: E402
from neraium.signals import generate_signals  # noqa: E402
from neraium.structural import build_structural_snapshot  # noqa: E402
from neraium.validation import validate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neraium Markets Day 5 validation pipeline"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save Day 5 output CSV/JSON artifacts under output/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1-9: existing Day 1-4 pipeline
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

    # 10-14: Day 5 validation layer
    evaluated = compute_forward_returns(signals, price_col="spy", horizons=[1, 5, 10])
    evaluated = score_action_usefulness(evaluated)
    calibration = evaluate_confidence_calibration(evaluated)
    baseline_comparison = compare_to_baselines(evaluated)
    summary = build_validation_report(evaluated)

    print("Total signals:", summary["total_signals"])
    print("\nCounts by regime:")
    for k, v in summary["regime_counts"].items():
        print(f"  {k}: {v}")

    print("\nCounts by action posture:")
    for k, v in summary["action_counts"].items():
        print(f"  {k}: {v}")

    print("\nAverage usefulness by horizon:")
    for k, v in summary["usefulness_summary"].items():
        print(f"  {k}: {v:.4f}")

    monotonic = bool(calibration["monotonic_non_decreasing"].iloc[0]) if not calibration.empty else False
    print("\nHigher confidence improved usefulness (monotonic bin check):", monotonic)

    print("\nBaseline comparison summary (avg_usefulness):")
    pivot = baseline_comparison.pivot(index="model", columns="horizon", values="avg_usefulness")
    print(pivot.round(4).to_string())

    if args.save_output:
        paths = save_validation_outputs(
            signals_df=evaluated,
            calibration_df=calibration,
            baseline_df=baseline_comparison,
            summary=summary,
            output_dir=_ROOT / "output",
        )
        print("\nSaved outputs:")
        for key, path in paths.items():
            print(f"  {key}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
