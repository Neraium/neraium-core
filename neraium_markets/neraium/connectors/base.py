"""Day 13: abstract connector interface for market data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class MarketDataConnector(ABC):
    """Read-only connector: load raw-ish frames; normalization happens downstream."""

    @abstractmethod
    def get_name(self) -> str:
        """Stable connector identifier (e.g. ``csv``, ``mock_api``)."""

    @abstractmethod
    def load_asset_data(self, asset: str, timeframe: str | None = None) -> pd.DataFrame:
        """Return a DataFrame for one asset (columns may vary by source)."""

    def load_many_assets(self, assets: list[str], timeframe: str | None = None) -> dict[str, pd.DataFrame]:
        """Load multiple assets; keys are lowercased asset names."""
        out: dict[str, pd.DataFrame] = {}
        for a in assets:
            key = a.strip().lower()
            out[key] = self.load_asset_data(key, timeframe=timeframe)
        return out

    @abstractmethod
    def get_source_metadata(self) -> dict[str, Any]:
        """Audit metadata (source id, retrieval hints, optional timing)."""
