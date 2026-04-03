"""Rolling in-memory live event buffer."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime

from neraium_core.markets.integrations.massive.models import NormalizedMarketEvent


class LiveBuffer:
    def __init__(self, retention: int = 1000) -> None:
        self.retention = retention
        self._events: dict[str, deque[NormalizedMarketEvent]] = defaultdict(lambda: deque(maxlen=self.retention))

    def add(self, event: NormalizedMarketEvent) -> None:
        self._events[event.ticker].append(event)

    def get_events(self, ticker: str, limit: int = 200) -> list[dict]:
        return [asdict(evt) for evt in list(self._events[ticker.upper()])[-limit:]]

    def latest_timestamp(self) -> datetime | None:
        latest = None
        for queue in self._events.values():
            if queue and (latest is None or queue[-1].timestamp > latest):
                latest = queue[-1].timestamp
        return latest

    def synchronized_latest(self, symbols: list[str]) -> dict[str, float] | None:
        frame: dict[str, float] = {}
        for symbol in symbols:
            queue = self._events.get(symbol.upper())
            if not queue:
                return None
            frame[symbol.upper()] = queue[-1].last
        return frame

    def symbol_count(self) -> int:
        return len(self._events)
