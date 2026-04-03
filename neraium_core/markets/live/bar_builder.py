"""Aggregate normalized live events into rolling bars."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from neraium_core.markets.integrations.massive.models import NormalizedBar, NormalizedMarketEvent


TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


class RollingBarBuilder:
    def __init__(self, timeframe: str = "1m", retention: int = 500) -> None:
        if timeframe not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported live timeframe: {timeframe}")
        self.timeframe = timeframe
        self.retention = retention
        self._bars: dict[str, deque[NormalizedBar]] = defaultdict(lambda: deque(maxlen=retention))

    def _bucket_start(self, ts: datetime) -> datetime:
        ts = ts.astimezone(timezone.utc)
        epoch = int(ts.timestamp())
        span = TIMEFRAME_SECONDS[self.timeframe]
        return datetime.fromtimestamp(epoch - (epoch % span), tz=timezone.utc)

    def add_event(self, event: NormalizedMarketEvent) -> NormalizedBar:
        ticker = event.ticker
        bucket = self._bucket_start(event.timestamp)
        bars = self._bars[ticker]
        if bars and bars[-1].timestamp == bucket:
            current = bars[-1]
            current.high = max(current.high, event.last)
            current.low = min(current.low, event.last)
            current.close = event.last
            current.volume += event.volume
            return current
        new_bar = NormalizedBar(
            timestamp=bucket,
            ticker=ticker,
            open=event.last,
            high=event.last,
            low=event.last,
            close=event.last,
            volume=event.volume,
            source=event.source,
            timeframe=self.timeframe,
        )
        bars.append(new_bar)
        return new_bar

    def get_bars(self, ticker: str, limit: int = 200) -> list[dict]:
        return [asdict(bar) for bar in list(self._bars[ticker.upper()])[-limit:]]

    def latest_closes(self, symbols: list[str], *, required_bars: int = 20) -> dict[str, list[float]] | None:
        out: dict[str, list[float]] = {}
        for symbol in symbols:
            bars = self._bars.get(symbol.upper())
            if bars is None or len(bars) < required_bars:
                return None
            out[symbol.upper()] = [bar.close for bar in list(bars)[-required_bars:]]
        return out
