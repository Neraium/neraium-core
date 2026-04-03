"""Massive payload adapters + normalized market schema."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class NormalizedMarketEvent:
    timestamp: datetime
    ticker: str
    last: float
    bid: float | None
    ask: float | None
    volume: float
    source: str
    event_type: str
    timeframe_hint: str


@dataclass(slots=True)
class NormalizedBar:
    timestamp: datetime
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    timeframe: str


def epoch_ms_to_dt(value: int | float) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def from_massive_agg(payload: dict, *, timeframe: str, source: str = "massive") -> NormalizedBar:
    return NormalizedBar(
        timestamp=epoch_ms_to_dt(payload["t"]),
        ticker=str(payload["T"]).upper(),
        open=float(payload["o"]),
        high=float(payload["h"]),
        low=float(payload["l"]),
        close=float(payload["c"]),
        volume=float(payload.get("v", 0.0)),
        source=source,
        timeframe=timeframe,
    )


def event_from_trade(payload: dict, *, timeframe_hint: str = "1m", source: str = "massive") -> NormalizedMarketEvent:
    return NormalizedMarketEvent(
        timestamp=epoch_ms_to_dt(payload.get("t") or payload.get("sip_timestamp") or payload["timestamp"]),
        ticker=str(payload.get("sym") or payload.get("T") or payload["ticker"]).upper(),
        last=float(payload.get("p") or payload.get("price") or payload.get("last", 0.0)),
        bid=float(payload["bp"]) if payload.get("bp") is not None else None,
        ask=float(payload["ap"]) if payload.get("ap") is not None else None,
        volume=float(payload.get("s") or payload.get("size") or payload.get("v") or 0.0),
        source=source,
        event_type=str(payload.get("ev") or payload.get("event", "trade")),
        timeframe_hint=timeframe_hint,
    )


def event_from_agg(payload: dict, *, timeframe_hint: str, source: str = "massive") -> NormalizedMarketEvent:
    return NormalizedMarketEvent(
        timestamp=epoch_ms_to_dt(payload["s"] if "s" in payload else payload["t"]),
        ticker=str(payload.get("sym") or payload.get("T") or payload["ticker"]).upper(),
        last=float(payload.get("c") or payload.get("close") or payload.get("last", 0.0)),
        bid=None,
        ask=None,
        volume=float(payload.get("v") or payload.get("volume") or 0.0),
        source=source,
        event_type=str(payload.get("ev") or payload.get("event", "aggregate")),
        timeframe_hint=timeframe_hint,
    )
