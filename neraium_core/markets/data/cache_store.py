"""CSV cache utilities for historical market data."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from neraium_core.markets.integrations.massive.models import NormalizedBar


DEFAULT_CACHE_PATH = "artifacts/neraium_markets/cache"


@dataclass(slots=True)
class CacheDataset:
    dataset_id: str
    provider: str
    symbols: list[str]
    timeframe: str
    start_date: str
    end_date: str
    created_at: str
    dataset_path: str


class CacheStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or os.getenv("NERAIUM_MARKETS_CACHE_PATH", DEFAULT_CACHE_PATH))
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "datasets.json"
        if not self.meta_path.exists():
            self.meta_path.write_text("[]", encoding="utf-8")

    def write_symbol_bars(self, dataset_id: str, timeframe: str, symbol: str, bars: list[NormalizedBar]) -> Path:
        dataset_dir = self.root / dataset_id / timeframe
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / f"{symbol.upper()}.csv"
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "open", "high", "low", "close", "volume", "source", "timeframe"])
            writer.writeheader()
            for bar in bars:
                writer.writerow(
                    {
                        "timestamp": bar.timestamp.isoformat(),
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                        "source": bar.source,
                        "timeframe": bar.timeframe,
                    }
                )
        return target

    def register_dataset(self, *, provider: str, symbols: list[str], timeframe: str, start_date: str, end_date: str, dataset_path: Path) -> CacheDataset:
        dataset = CacheDataset(
            dataset_id=dataset_path.name,
            provider=provider,
            symbols=[s.upper() for s in symbols],
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset_path=str(dataset_path),
        )
        all_datasets = self.list_datasets()
        all_datasets.append(dataset)
        self.meta_path.write_text(json.dumps([asdict(item) for item in all_datasets], indent=2), encoding="utf-8")
        return dataset

    def list_datasets(self) -> list[CacheDataset]:
        raw = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return [CacheDataset(**item) for item in raw]
