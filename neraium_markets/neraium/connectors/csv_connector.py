"""Day 13: CSV files on disk (Day 1 schema), optional timeframe subfolders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from neraium.connectors.base import MarketDataConnector


class CSVConnector(MarketDataConnector):
    """Load ``{asset}.csv`` from ``data_dir`` or ``data_dir/{timeframe}/`` when present."""

    def __init__(self, data_dir: Path, *, source_label: str = "csv") -> None:
        self._data_dir = Path(data_dir)
        self._source_label = source_label

    def get_name(self) -> str:
        return self._source_label

    def _resolve_path(self, asset: str, timeframe: str | None) -> Path:
        key = asset.strip().lower()
        if timeframe:
            sub = self._data_dir / timeframe / f"{key}.csv"
            if sub.is_file():
                return sub
        return self._data_dir / f"{key}.csv"

    def load_asset_data(self, asset: str, timeframe: str | None = None) -> pd.DataFrame:
        path = self._resolve_path(asset, timeframe)
        if not path.is_file():
            raise FileNotFoundError(f"CSVConnector: missing file for {asset!r}: {path}")
        return pd.read_csv(path)

    def get_source_metadata(self) -> dict[str, Any]:
        return {
            "source_name": self._source_label,
            "data_dir": str(self._data_dir.resolve()),
            "connector": "CSVConnector",
        }
