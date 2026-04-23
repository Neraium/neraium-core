"""Day 13: simulated API responses from local JSON fixtures (no live HTTP)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from neraium.connectors.base import MarketDataConnector


class MockAPIConnector(MarketDataConnector):
    """
    Pretend to call an API: read ``{fixture_dir}/{asset}.json``, optional latency sleep.

    JSON is either a list of row objects or ``{"bars": [...]}``.
    """

    def __init__(
        self,
        fixture_dir: Path,
        *,
        simulated_latency_ms: float = 2.0,
        source_name: str = "mock_api",
    ) -> None:
        self._fixture_dir = Path(fixture_dir)
        self._simulated_latency_ms = float(simulated_latency_ms)
        self._source_name = source_name
        self._retrieval_ts = datetime.now(timezone.utc).isoformat()

    def get_name(self) -> str:
        return self._source_name

    def _load_payload(self, asset: str) -> list[dict[str, Any]]:
        key = asset.strip().lower()
        path = self._fixture_dir / f"{key}.json"
        if not path.is_file():
            raise FileNotFoundError(f"MockAPIConnector: missing fixture {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "bars" in raw:
            return list(raw["bars"])
        if isinstance(raw, list):
            return raw
        raise ValueError(f"MockAPIConnector: unexpected JSON shape in {path}")

    def load_asset_data(self, asset: str, timeframe: str | None = None) -> pd.DataFrame:
        if self._simulated_latency_ms > 0:
            time.sleep(self._simulated_latency_ms / 1000.0)
        rows = self._load_payload(asset)
        return pd.DataFrame(rows)

    def get_source_metadata(self) -> dict[str, Any]:
        return {
            "source_name": self._source_name,
            "retrieval_timestamp": self._retrieval_ts,
            "simulated_latency_ms": self._simulated_latency_ms,
            "fixture_dir": str(self._fixture_dir.resolve()),
            "connector": "MockAPIConnector",
        }
