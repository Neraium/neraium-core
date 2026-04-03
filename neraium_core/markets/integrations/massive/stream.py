"""Massive websocket streaming adapter."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from .config import MassiveConfig
from .models import NormalizedMarketEvent, event_from_agg, event_from_trade

try:
    import websockets
except Exception:  # noqa: BLE001
    websockets = None


class MassiveStreamClient:
    def __init__(self, config: MassiveConfig, *, timeframe_hint: str = "1m") -> None:
        self.config = config
        self.timeframe_hint = timeframe_hint
        self._socket = None
        self._symbols: list[str] = []

    async def connect(self) -> None:
        self.config.validate(require_api_key=True)
        if websockets is None:
            raise RuntimeError("websockets dependency is required for live Massive streaming")
        self._socket = await websockets.connect(self.config.ws_base_url)
        await self._socket.send(json.dumps({"action": "auth", "params": self.config.api_key}))

    async def close(self) -> None:
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def subscribe(self, symbols: list[str]) -> None:
        self._symbols = [s.upper() for s in symbols]
        if self._socket is None:
            return
        channels = ",".join(f"A.{symbol}" for symbol in self._symbols)
        await self._socket.send(json.dumps({"action": "subscribe", "params": channels}))

    async def unsubscribe_all(self) -> None:
        if self._socket is None or not self._symbols:
            return
        channels = ",".join(f"A.{symbol}" for symbol in self._symbols)
        await self._socket.send(json.dumps({"action": "unsubscribe", "params": channels}))

    async def iter_events(self) -> AsyncIterator[NormalizedMarketEvent]:
        if self._socket is None:
            raise RuntimeError("stream not connected")
        while True:
            raw_msg = await self._socket.recv()
            payload = json.loads(raw_msg)
            rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                ev = row.get("ev")
                if ev in {"A", "AM"}:
                    yield event_from_agg(row, timeframe_hint=self.timeframe_hint)
                elif ev in {"T", "XT"}:
                    yield event_from_trade(row, timeframe_hint=self.timeframe_hint)


async def stream_with_reconnect(client: MassiveStreamClient, *, symbols: list[str], reconnect_delay: float = 2.0) -> AsyncIterator[NormalizedMarketEvent]:
    while True:
        try:
            await client.connect()
            await client.subscribe(symbols)
            async for event in client.iter_events():
                yield event
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            await asyncio.sleep(reconnect_delay)
        finally:
            await client.close()
