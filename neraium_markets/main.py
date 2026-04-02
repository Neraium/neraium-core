#!/usr/bin/env python3
"""Neraium Markets: load → validate → align → features → structural → signals (Day 4)."""

from __future__ import annotations

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


def main() -> int:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
