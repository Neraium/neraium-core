#!/usr/bin/env python3
"""Neraium Markets: load → validate → align → features → structural → signals (Day 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.alignment import align_close_series  # noqa: E402
from neraium.data_loader import load_all_assets  # noqa: E402
from neraium.features import build_feature_table  # noqa: E402
from neraium.signals import generate_signals, save_signals_csv  # noqa: E402
from neraium.structural import build_structural_snapshot  # noqa: E402
from neraium.validation import validate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Neraium Markets pipeline: features, structural snapshot, regime signals"
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Also save output/features.csv and output/structural_snapshot.csv",
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

    print("Aligned closes shape:", merged.shape)
    print("Feature table shape:", features.shape)
    print("Structural snapshot shape:", structural.shape)
    print("Signals shape:", signals.shape)
    print()
    print("Regime distribution:")
    print(signals["regime_label"].value_counts().to_string())
    print()
    print("Last 10 signal rows:")
    print(signals.tail(10).to_string(index=False))

    out_path = save_signals_csv(signals)
    print()
    print("Wrote:", out_path.resolve())

    if args.save_output:
        out_dir = _ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_dir / "features.csv", index=False)
        structural.to_csv(out_dir / "structural_snapshot.csv", index=False)
        print("Also saved:", out_dir / "features.csv")
        print("Also saved:", out_dir / "structural_snapshot.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
