"""Historical bar fetching and normalization for Massive."""

from __future__ import annotations

from dataclasses import dataclass

from neraium_core.markets.data.cache_store import CacheStore

from .client import MassiveRestClient
from .models import NormalizedBar, from_massive_agg

TIMEFRAME_TO_RANGE = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "1h": (1, "hour"),
    "daily": (1, "day"),
}
DEFAULT_BASKET = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT"]


@dataclass(slots=True)
class MassiveHistoricalService:
    client: MassiveRestClient

    def fetch_bars(self, *, symbol: str, timeframe: str, start_date: str, end_date: str) -> list[NormalizedBar]:
        if timeframe not in TIMEFRAME_TO_RANGE:
            raise ValueError(f"Unsupported timeframe {timeframe}; choose from {sorted(TIMEFRAME_TO_RANGE)}")
        multiplier, span = TIMEFRAME_TO_RANGE[timeframe]
        path = f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{span}/{start_date}/{end_date}"
        response = self.client.get_json(path, params={"adjusted": "true", "sort": "asc", "limit": "50000"})
        raw_results = response.get("results") or []
        return [from_massive_agg({**row, "T": symbol.upper()}, timeframe=timeframe) for row in raw_results]

    def fetch_and_cache(
        self,
        *,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        cache_store: CacheStore,
        dataset_id: str,
    ) -> dict:
        fetched: dict[str, int] = {}
        for symbol in symbols:
            bars = self.fetch_bars(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
            if not bars:
                continue
            cache_store.write_symbol_bars(dataset_id, timeframe, symbol, bars)
            fetched[symbol.upper()] = len(bars)

        dataset_path = cache_store.root / dataset_id
        meta = cache_store.register_dataset(
            provider="massive",
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            dataset_path=dataset_path,
        )
        return {"dataset": meta, "fetched_rows": fetched}
