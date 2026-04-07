"""Orchestrates live market-data ingestion into existing markets signal pipeline."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from neraium_core.markets.data.validation import validate_market_frame
from neraium_core.markets.defaults import DEFAULT_LIVE_TIMEFRAME
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.integrations.massive.config import load_massive_config
from neraium_core.markets.integrations.massive.stream import MassiveStreamClient, stream_with_reconnect
from neraium_core.markets.persistence.sqlite_store import MarketsSQLiteStore
from neraium_core.markets.signals.emission_control import EmissionControlConfig, SignalEmissionController
from neraium_core.markets.signals.signal_generator import generate_signal_for_asset
from neraium_core.markets.state.state_vector import CORE, build_state_vector

from .bar_builder import RollingBarBuilder
from .live_buffer import LiveBuffer
from .session_state import LiveSessionState

TIMEFRAME_FREQ = {"1m": "min", "5m": "5min", "15m": "15min"}


class LiveSessionRunner:
    def __init__(self) -> None:
        self.state = LiveSessionState.DISCONNECTED
        self.symbols: list[str] = []
        self.timeframe = DEFAULT_LIVE_TIMEFRAME
        self.buffer = LiveBuffer(retention=int(os.getenv("NERAIUM_LIVE_EVENT_RETENTION", "5000")))
        self.bars = RollingBarBuilder(timeframe=DEFAULT_LIVE_TIMEFRAME, retention=int(os.getenv("NERAIUM_LIVE_BAR_RETENTION", "2000")))
        self.store = MarketsSQLiteStore()
        self.evidence = EvidenceLog(path="artifacts/neraium_markets/live_evidence.jsonl")
        self.controller = SignalEmissionController(EmissionControlConfig())
        self._task: asyncio.Task | None = None
        self.last_signal_at: datetime | None = None
        self.latest_error: str | None = None
        self.latest_signals: dict[str, dict] = {}
        self.suppressed_count = 0
        self.abstain_count = 0
        self.reconnect_count = 0
        self.last_suppressed_at: datetime | None = None
        self.last_suppressed_reason: str | None = None
        self.warmup_bars_required = int(os.getenv("NERAIUM_LIVE_WARMUP_BARS", "30"))
        self.data_stale_after_seconds = int(os.getenv("NERAIUM_LIVE_DATA_STALE_SECONDS", "90"))

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
        if timeframe not in TIMEFRAME_FREQ:
            raise ValueError("Live timeframe must be one of 1m, 5m, 15m")
        self.timeframe = timeframe
        self.bars = RollingBarBuilder(timeframe=timeframe)
        self.controller = SignalEmissionController(EmissionControlConfig(warmup_bars=self.warmup_bars_required))
        self.latest_error = None
        self.latest_signals = {}
        self.suppressed_count = 0
        self.abstain_count = 0
        self.last_suppressed_at = None
        self.last_suppressed_reason = None
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

    def _bars_collected(self) -> int:
        closes = self.bars.latest_closes(self.symbols, required_bars=1)
        if closes is None or not closes:
            return 0
        counts = [len(values) for values in closes.values() if values is not None]
        if not counts or any(count == 0 for count in counts):
            return 0
        return min(counts, default=0)

    def _warmup_metrics(self) -> tuple[int, int, float]:
        collected = self._bars_collected()
        required = self.warmup_bars_required
        pct = 1.0 if required <= 0 else min(1.0, collected / required)
        return collected, required, pct

    def _readiness_state(self) -> str:
        collected, required, _ = self._warmup_metrics()
        last_event = self.buffer.latest_timestamp()
        if self.state in {LiveSessionState.DISCONNECTED, LiveSessionState.STOPPED}:
            return "disconnected"
        if self.state == LiveSessionState.ERROR:
            return "error"
        if self.state == LiveSessionState.CONNECTING:
            return "connecting"
        if self.state == LiveSessionState.RECONNECTING:
            return "reconnecting"
        if self.buffer.symbol_count() == 0:
            return "connected_no_data"
        if last_event and datetime.now(timezone.utc) - last_event > timedelta(seconds=self.data_stale_after_seconds):
            return "connected_no_data"
        if collected < required:
            return "receiving_data_warming_up"
        if self.last_signal_at is None:
            return "live_no_valid_signals"
        return "live_ready"

    async def _run(self) -> None:
        cfg = load_massive_config()
        client = MassiveStreamClient(cfg, timeframe_hint=self.timeframe)
        frame_index = 0

        def _mark_reconnect(exc: Exception) -> None:
            self.state = LiveSessionState.RECONNECTING
            self.latest_error = f"Live feed reconnecting after provider error: {exc}"
            self.store.record_error({"error": self.latest_error})

        try:
            async for event in stream_with_reconnect(client, symbols=self.symbols, on_reconnect=_mark_reconnect):
                if self.state == LiveSessionState.RECONNECTING:
                    self.reconnect_count += 1
                if self.state in {LiveSessionState.CONNECTING, LiveSessionState.RECONNECTING}:
                    self.state = LiveSessionState.CONNECTED_WARMING_UP
                self.buffer.add(event)
                self.store.record_live_event(event.ticker, {**asdict(event), "timestamp": event.timestamp.isoformat()}, self.buffer.retention)
                bar = self.bars.add_event(event)
                self.store.record_live_bar(bar.ticker, {**asdict(bar), "timestamp": bar.timestamp.isoformat()}, self.bars.retention)

                closes = self.bars.latest_closes(self.symbols, required_bars=max(30, self.warmup_bars_required))
                if closes is None:
                    continue
                frame_index += 1
                prices = pd.DataFrame(closes)
                prices.index = pd.date_range(end=bar.timestamp, periods=len(prices), freq=TIMEFRAME_FREQ[self.timeframe])
                state = build_state_vector(prices)
                validation = validate_market_frame(prices)

                if self._bars_collected() >= self.warmup_bars_required:
                    self.state = LiveSessionState.CONNECTED_LIVE

                for asset in self.symbols:
                    signal = generate_signal_for_asset(asset=asset, timeframe=self.timeframe, state=state, data_quality=float(validation["completeness"]))
                    if signal is None or signal.trader_output is None or signal.decision_validity is None:
                        continue

                    stable_permission = self.controller.stabilize_permission(
                        f"{asset}:{self.timeframe}",
                        signal.trader_output.action_permission,
                        signal.trader_output.validity_score,
                    )
                    signal.trader_output.action_permission = stable_permission
                    signal.trader_output.trading_signal = f"{stable_permission}:{signal.trader_output.best_action}"
                    signal.decision_validity.action_permission = stable_permission
                    signal.trader_output.timestamp = bar.timestamp
                    signal.timestamp = bar.timestamp

                    decision = self.controller.evaluate(
                        f"{asset}:{self.timeframe}",
                        frame_index=frame_index,
                        sample_count=len(prices),
                        confidence=signal.trader_output.confidence,
                        action_permission=signal.trader_output.action_permission,
                        best_action=signal.trader_output.best_action,
                        validity_score=signal.trader_output.validity_score,
                        market_state=signal.trader_output.market_state,
                        transition_state=signal.trader_output.transition_state,
                        abstain=signal.decision_validity.abstain,
                    )
                    signal.trader_output.cooldown_status = decision.cooldown_status
                    signal_json = signal.trader_output.model_dump(mode="json")
                    signal_json["emitted"] = decision.emit

                    if signal.decision_validity.abstain:
                        self.abstain_count += 1

                    if not decision.emit:
                        self.suppressed_count += 1
                        self.last_suppressed_at = datetime.now(timezone.utc)
                        self.last_suppressed_reason = decision.suppression_status
                        self.store.record_suppressed_output(asset, signal_json)
                        continue

                    self.controller.mark_emitted(
                        f"{asset}:{self.timeframe}",
                        frame_index=frame_index,
                        action_permission=signal.trader_output.action_permission,
                        best_action=signal.trader_output.best_action,
                        validity_score=signal.trader_output.validity_score,
                        market_state=signal.trader_output.market_state,
                        transition_state=signal.trader_output.transition_state,
                    )
                    self.latest_signals[asset] = signal_json
                    self.last_signal_at = datetime.now(timezone.utc)
                    self.store.record_trader_output(asset, signal_json)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.state = LiveSessionState.ERROR
            if "MASSIVE_API_KEY" in str(exc):
                self.latest_error = "Missing API key for Massive provider (MASSIVE_API_KEY)"
            else:
                self.latest_error = f"Provider unavailable or live startup failed: {exc}"
            self.store.record_error({"error": self.latest_error})

    def status(self) -> dict:
        bars_collected, bars_required, warmup_progress = self._warmup_metrics()
        return {
            "running": bool(self._task and not self._task.done()),
            "session_state": self.state.value,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "warmup_progress": warmup_progress,
            "warmup_progress_pct": round(warmup_progress * 100.0, 2),
            "bars_collected": bars_collected,
            "bars_required": bars_required,
            "readiness_state": self._readiness_state(),
            "buffered_symbol_count": self.buffer.symbol_count(),
            "last_event_at": self.buffer.latest_timestamp().isoformat() if self.buffer.latest_timestamp() else None,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "latest_error": self.latest_error,
            "suppressed_count": self.suppressed_count,
            "abstain_count": self.abstain_count,
            "reconnect_count": self.reconnect_count,
            "last_suppressed_at": self.last_suppressed_at.isoformat() if self.last_suppressed_at else None,
            "last_suppressed_reason": self.last_suppressed_reason,
        }
