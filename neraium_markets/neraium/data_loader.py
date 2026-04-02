from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import ASSETS, DATA_DIR, DATE_COLUMN


def load_asset_csv(asset: str) -> pd.DataFrame:
    """Load one asset CSV from sample_data and return a normalized dataframe."""
    csv_path = Path(DATA_DIR) / f"{asset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV for asset '{asset}': {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if DATE_COLUMN not in df.columns:
        raise ValueError(f"CSV '{asset}' missing required date column '{DATE_COLUMN}'")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
    return df


def load_all_assets() -> dict[str, pd.DataFrame]:
    """Load all configured assets as a mapping from symbol to dataframe."""
    return {asset: load_asset_csv(asset) for asset in ASSETS}
