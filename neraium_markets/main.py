#!/usr/bin/env python3
"""Neraium Markets pipeline entrypoint (Day 3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root (this directory) is importable
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATE_COLUMN  # noqa: E402
from neraium.alignment import align_close_series  # noqa: E402
from neraium.data_loader import load_all_assets  # noqa: E402
from neraium.features import build_feature_table  # noqa: E402
from neraium.structure import build_structural_snapshot  # noqa: E402
from neraium.validation import validate_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neraium Markets Day 3 structural pipeline")
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save output/features.csv and output/structural_snapshot.csv",
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

    aligned = align_close_series(data)
    features = build_feature_table(aligned)
    snapshot = build_structural_snapshot(features)

    print("Aligned dataset shape:", aligned.shape)
    print("Feature dataset shape:", features.shape)
    print("Structural snapshot shape:", snapshot.shape)

    preview_cols = [
        DATE_COLUMN,
        "corr_drift_score",
        "lag_drift_score",
        "sector_entropy",
        "sector_concentration_score",
        "instability_score",
        "coherence_score",
    ]
    present_preview_cols = [col for col in preview_cols if col in snapshot.columns]
    print(snapshot[present_preview_cols].head(10).to_string(index=False))

    if args.save_output:
        out_dir = _ROOT / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(out_dir / "features.csv", index=False)
        snapshot.to_csv(out_dir / "structural_snapshot.csv", index=False)
        print(f"Saved outputs to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
