"""Load market CSVs into pandas DataFrames."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow `python main.py` from neraium_markets without installing the package.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


def _data_dir(override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    return _ROOT / config.DATA_DIR


def load_asset_csv(asset: str, *, data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load a single asset CSV from sample_data.

    Columns are lowercased, timestamp parsed, rows sorted ascending by time.
    """
    key = asset.strip().lower()
    path = _data_dir(data_dir) / f"{key}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV for asset {asset!r}: {path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if config.DATE_COLUMN not in df.columns:
        raise ValueError(f"Expected column {config.DATE_COLUMN!r} in {path}")

    df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN], utc=False)
    df = df.sort_values(config.DATE_COLUMN, ascending=True).reset_index(drop=True)
    return df


def load_all_assets(
    *,
    data_dir: Path | None = None,
    assets: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load assets from ``config.ASSETS`` (or ``assets`` if provided) under ``data_dir`` or default ``sample_data``."""
    names = list(assets) if assets is not None else list(config.ASSETS)
    out: dict[str, pd.DataFrame] = {}
    for asset in names:
        out[asset.lower()] = load_asset_csv(asset, data_dir=data_dir)
    return out
