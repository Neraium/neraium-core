from __future__ import annotations

import sys
from pathlib import Path

from config import DATE_COLUMN
from neraium.alignment import align_close_series
from neraium.data_loader import load_all_assets
from neraium.features import build_feature_table
from neraium.validation import validate_all


def main() -> int:
    data = load_all_assets()
    errors = validate_all(data)

    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    aligned = align_close_series(data)
    features = build_feature_table(aligned)

    print(f"Aligned shape: {aligned.shape}")
    print(f"Feature shape: {features.shape}")

    sample_cols = [
        DATE_COLUMN,
        "spy_ret_1d",
        "spy_ret_5d",
        "spy_vol_10d",
        "breadth_pct_above_20dma",
        "sector_dispersion_1d",
        "sector_concentration_top2",
        "rates_2s10s",
        "risk_off_proxy",
    ]
    sample_cols = [c for c in sample_cols if c in features.columns]

    print("\nSample feature columns:")
    print(features[sample_cols].head(10).to_string(index=False))

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "features.csv"
    features.to_csv(out_path, index=False)
    print(f"\nSaved feature table to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
