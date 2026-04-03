"""Orchestrates live market-data ingestion into existing markets signal pipeline."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from neraium_core.markets.data.validation import validate_market_frame
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.integrations.massive.config import load_massive_config
from neraium_core.markets.integrations.massive.stream import MassiveStreamClient, stream_with_reconnect
from neraium_core.markets.persistence.sqlite_store import MarketsSQLiteStore
from neraium_core.markets.signals.signal_generator import generate_signal_for_asset
from neraium_core.markets.state.state_vector import CORE, build_state_vector

from .bar_builder import RollingBarBuilder
from .live_buffer import LiveBuffer
from .session_state import LiveSessionState


class LiveSessionRunner:
    def __init__(self) -> None:
        self.state = LiveSessionState.DISCONNECTED
        self.symbols: list[str] = []
        self.timeframe = "1m"
        self.buffer = LiveBuffer(retention=int(os.getenv("NERAIUM_LIVE_EVENT_RETENTION", "5000")))
        self.bars = RollingBarBuilder(timeframe="1m", retention=int(os.getenv("NERAIUM_LIVE_BAR_RETENTION", "2000")))
        self.store = MarketsSQLiteStore()
        self.evidence = EvidenceLog(path="artifacts/neraium_markets/live_evidence.jsonl")
        self._task: asyncio.Task | None = None
        self.last_signal_at: datetime | None = None
        self.latest_error: str | None = None
        self.latest_signals: dict[str, dict] = {}

    async def start(self, symbols: list[str], timeframe: str) -> None:
        if self._task and not self._task.done():
            return
        self.symbols = [s.upper() for s in symbols]
        missing_required = [sym for sym in CORE if sym not in set(self.symbols)]
        if missing_required:
            raise ValueError(
                "Live session requires core symbols for state vector construction. "
                f"Missing required symbols: {', '.join(missing_required)}"
            )
        self.timeframe = timeframe
        self.bars = RollingBarBuilder(timeframe=timeframe)
        self.state = LiveSessionState.CONNECTING
        self.store.record_live_session({"state": self.state.value, "symbols": self.symbols, "timeframe": timeframe})
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        self.state = LiveSessionState.STOPPED

    async def _run(self) -> None:
        cfg = load_massive_config()
        client = MassiveStreamClient(cfg, timeframe_hint=self.timeframe)
        try:
            async for event in stream_with_reconnect(client, symbols=self.symbols):
                if self.state in {LiveSessionState.CONNECTING, LiveSessionState.RECONNECTING}:
                    self.state = LiveSessionState.CONNECTED_WARMING_UP
                self.buffer.add(event)
                self.store.record_live_event(event.ticker, {**asdict(event), "timestamp": event.timestamp.isoformat()}, self.buffer.retention)
                bar = self.bars.add_event(event)
                self.store.record_live_bar(bar.ticker, {**asdict(bar), "timestamp": bar.timestamp.isoformat()}, self.bars.retention)
                closes = self.bars.latest_closes(self.symbols, required_bars=30)
                if closes is None:
                    continue
                self.state = LiveSessionState.CONNECTED_LIVE
                prices = pd.DataFrame(closes)
                prices.index = pd.date_range(end=datetime.now(timezone.utc), periods=len(prices), freq="min")
                state = build_state_vector(prices)
                validation = validate_market_frame(prices)
                for asset in self.symbols:
                    signal = generate_signal_for_asset(asset=asset, timeframe=self.timeframe, state=state, data_quality=float(validation["completeness"]))
                    if signal is None or signal.trader_output is None:
                        continue
                    signal_json = signal.trader_output.model_dump(mode="json")
                    self.latest_signals[asset] = signal_json
                    self.last_signal_at = datetime.now(timezone.utc)
                    self.store.record_trader_output(asset, signal_json)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.state = LiveSessionState.ERROR
            self.latest_error = str(exc)
            self.store.record_error({"error": self.latest_error})

    def status(self) -> dict:
        return {
            "running": bool(self._task and not self._task.done()),
            "session_state": self.state.value,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "warmup_progress": 1.0 if self.state == LiveSessionState.CONNECTED_LIVE else 0.0,
            "buffered_symbol_count": self.buffer.symbol_count(),
            "last_event_at": self.buffer.latest_timestamp().isoformat() if self.buffer.latest_timestamp() else None,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "latest_error": self.latest_error,
        }
