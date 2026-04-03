"""Historical ingestion workflow for Massive + local CSV cache."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

from neraium_core.markets.data.cache_store import CacheStore
from neraium_core.markets.integrations.massive.client import MassiveRestClient
from neraium_core.markets.integrations.massive.config import load_massive_config
from neraium_core.markets.integrations.massive.historical import DEFAULT_BASKET, MassiveHistoricalService


class HistoricalIngestService:
    def __init__(self, cache_store: CacheStore | None = None) -> None:
        self.cache_store = cache_store or CacheStore()

    def fetch_massive(self, *, symbols: list[str] | None, timeframe: str, start_date: str, end_date: str) -> dict:
        config = load_massive_config()
        config.validate(require_api_key=True)
        service = MassiveHistoricalService(MassiveRestClient(config))
        use_symbols = [s.upper() for s in (symbols or DEFAULT_BASKET)]
        dataset_id = datetime.now(timezone.utc).strftime("massive_%Y%m%dT%H%M%SZ")
        result = service.fetch_and_cache(
            symbols=use_symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            cache_store=self.cache_store,
            dataset_id=dataset_id,
        )
        result["dataset"] = asdict(result["dataset"])
        return result
