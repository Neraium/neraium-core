from __future__ import annotations

from pathlib import Path

from neraium_core.markets.data.cache_store import CacheStore
from neraium_core.markets.integrations.massive.historical import MassiveHistoricalService
from neraium_core.markets.integrations.massive.models import event_from_agg


class DummyClient:
    def get_json(self, path: str, params=None):
        return {
            "results": [
                {"t": 1704067200000, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5, "v": 1000},
                {"t": 1704067260000, "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5, "v": 1200},
            ]
        }


def test_historical_fetch_normalization(tmp_path: Path):
    service = MassiveHistoricalService(client=DummyClient())
    bars = service.fetch_bars(symbol="SPY", timeframe="1m", start_date="2026-01-01", end_date="2026-01-02")
    assert len(bars) == 2
    assert bars[0].ticker == "SPY"
    assert bars[1].close == 101.5


def test_cache_write_read(tmp_path: Path):
    service = MassiveHistoricalService(client=DummyClient())
    bars = service.fetch_bars(symbol="SPY", timeframe="1m", start_date="2026-01-01", end_date="2026-01-02")
    store = CacheStore(root=tmp_path)
    out = store.write_symbol_bars("ds1", "1m", "SPY", bars)
    assert out.exists()
    datasets = store.list_datasets()
    assert datasets == []


def test_live_payload_normalization():
    evt = event_from_agg({"ev": "A", "sym": "QQQ", "s": 1704067200000, "c": 350.2, "v": 4500}, timeframe_hint="1m")
    assert evt.ticker == "QQQ"
    assert evt.last == 350.2
