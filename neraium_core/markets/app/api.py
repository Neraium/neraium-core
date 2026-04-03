"""FastAPI endpoints for Neraium Markets."""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from neraium_core.markets.data.alignment import align_to_shared_clock
from neraium_core.markets.data.cache_store import CacheStore
from neraium_core.markets.data.historical_ingest import HistoricalIngestService
from neraium_core.markets.data.market_data_loader import MarketDataLoader
from neraium_core.markets.data.validation import validate_market_frame
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.integrations.massive.config import MassiveConfigError, load_massive_config
from neraium_core.markets.live.live_runner import LiveSessionRunner
from neraium_core.markets.persistence.sqlite_store import MarketsSQLiteStore
from neraium_core.markets.replay import run_signal_replay
from neraium_core.markets.signals.signal_generator import generate_signal_for_asset, to_evidence
from neraium_core.markets.state.state_vector import CORE, build_state_vector

TIMEFRAMES = ["daily", "1h", "15m"]
DEFAULT_LIVE_SYMBOLS = list(CORE)
LOGGER = logging.getLogger(__name__)


class HistoricalFetchBody(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA"])
    timeframe: str = "15m"
    start_date: str
    end_date: str


class LiveStartBody(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: DEFAULT_LIVE_SYMBOLS.copy())
    timeframe: str = "5m"


def run_signal_pipeline(data_dir: Path, evidence_log: EvidenceLog) -> list[dict]:
    loader = MarketDataLoader(data_dir=data_dir)
    generated: list[dict] = []
    for timeframe in TIMEFRAMES:
        prices = align_to_shared_clock(loader.load(timeframe))
        validation = validate_market_frame(prices)
        state = build_state_vector(prices)
        for asset in CORE:
            signal = generate_signal_for_asset(
                asset=asset,
                timeframe=timeframe,
                state=state,
                data_quality=float(validation["completeness"]),
            )
            if signal is None:
                continue
            evidence_log.append(to_evidence(signal, {"validation": validation, "timeframe": timeframe}))
            generated.append(signal.model_dump(mode="json"))
    return generated


def create_app(
    data_dir: str | Path = "neraium_core/markets/sample_data",
    evidence_path: str | Path = "artifacts/neraium_markets/evidence.jsonl",
) -> FastAPI:
    app = FastAPI(title="Neraium Markets API", version="0.2.0")
    evidence = EvidenceLog(path=evidence_path)
    cache_store = CacheStore()
    ingest = HistoricalIngestService(cache_store)
    sqlite = MarketsSQLiteStore()
    live = LiveSessionRunner()
    configured_data_dir = Path(data_dir)
    app.state.data_dir = configured_data_dir

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "neraium-markets"}

    @app.get("/")
    def operator_ui() -> HTMLResponse:
        html_path = Path(__file__).parent / "static" / "operator.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.post("/run-signals")
    def run_signals() -> dict[str, list[dict]]:
        signals = run_signal_pipeline(app.state.data_dir, evidence)
        return {"signals": signals}

    @app.get("/signals/latest")
    def latest_signals(limit: int = 20) -> list[dict]:
        return [sig.model_dump(mode="json") for sig in evidence.latest(limit)]

    @app.post("/integrations/massive/historical/fetch")
    def fetch_massive_historical(body: HistoricalFetchBody) -> dict:
        try:
            result = ingest.fetch_massive(symbols=body.symbols, timeframe=body.timeframe, start_date=body.start_date, end_date=body.end_date)
            sqlite.record_fetch_job(result)
            return result
        except MassiveConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/integrations/massive/status")
    def massive_status() -> dict:
        cfg = load_massive_config()
        latest_check = datetime.now(timezone.utc).isoformat()
        return {
            "provider_configured": bool(cfg.rest_base_url and cfg.ws_base_url),
            "api_key_present": bool(cfg.api_key),
            "latest_connectivity_check": latest_check,
            "live_capability": bool(cfg.api_key),
        }

    @app.post("/run-replay")
    def run_replay(
        request: Request,
        timeframe: str = "15m",
        data_dir: str | None = None,
        use_massive_cached_data: bool = False,
        symbols: str | None = None,
    ) -> dict[str, object]:
        replay_data_dir = Path(data_dir) if data_dir else getattr(request.app.state, "data_dir", None)
        if use_massive_cached_data:
            datasets = cache_store.list_datasets()
            if not datasets:
                raise HTTPException(status_code=400, detail="No cached Massive datasets found")
            replay_data_dir = Path(datasets[-1].dataset_path)

        if replay_data_dir is None:
            raise HTTPException(
                status_code=400,
                detail="No replay data_dir configured. Provide data_dir or configure create_app(data_dir=...).",
            )

        replay_frame_dir = replay_data_dir / timeframe
        if not replay_frame_dir.exists() or not any(replay_frame_dir.glob("*.csv")):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid replay data_dir '{replay_data_dir}': missing CSV data for timeframe '{timeframe}'.",
            )

        LOGGER.debug(
            "run_replay using data_dir=%s timeframe=%s use_massive_cached_data=%s",
            replay_data_dir,
            timeframe,
            use_massive_cached_data,
        )

        use_symbols = [s.strip().upper() for s in symbols.split(",")] if symbols else None
        run_id = f"replay_{uuid4().hex[:12]}"
        csv_output = Path("artifacts/neraium_markets/replay") / f"{run_id}.csv"
        rows = run_signal_replay(replay_data_dir, evidence, timeframe=timeframe, csv_output_path=csv_output, symbols=use_symbols)
        run_meta = {"timeframe": timeframe, "data_dir": str(replay_data_dir), "signal_count": len(rows), "csv_path": str(csv_output)}
        sqlite.record_replay_run(run_id, run_meta)
        for row in rows:
            sqlite.record_replay_output(run_id, row)
        return {"run_id": run_id, "replay": rows, "meta": run_meta}

    @app.post("/live/start")
    async def live_start(body: LiveStartBody) -> dict:
        effective_symbols = [item.upper() for item in (body.symbols or DEFAULT_LIVE_SYMBOLS)]
        missing_required = [sym for sym in CORE if sym not in set(effective_symbols)]
        if missing_required:
            required_symbols = ", ".join(CORE)
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Live session requires core symbols: {required_symbols}. "
                    + f"Missing: {', '.join(missing_required)}"
                ),
            )
        try:
            load_massive_config().validate(require_api_key=True)
        except MassiveConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            await live.start(effective_symbols, body.timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "started", **live.status()}

    @app.post("/live/stop")
    async def live_stop() -> dict:
        await live.stop()
        return {"status": "stopped", **live.status()}

    @app.get("/live/status")
    def live_status() -> dict:
        return live.status()

    @app.get("/live/signals/latest")
    def live_latest_signals() -> dict:
        return {"signals": list(live.latest_signals.values())}

    @app.get("/signals/history")
    def signal_history(limit: int = 300, ticker: str | None = None, session_type: str | None = None) -> dict[str, list[dict]]:
        ticker_upper = ticker.upper() if ticker else None
        rows: list[dict] = []
        include_live = session_type in (None, "live")
        include_replay = session_type in (None, "replay")
        if include_live:
            for row in sqlite.list_trader_outputs(limit=limit, ticker=ticker_upper):
                rows.append({**row, "session_type": "live"})
        if include_replay:
            for row in sqlite.list_replay_outputs(limit=limit, ticker=ticker_upper):
                rows.append({**row, "session_type": "replay"})
        rows.sort(key=lambda item: item.get("timestamp") or item.get("created_at") or "", reverse=True)
        return {"signals": rows[:limit]}

    @app.get("/integrations/massive/datasets")
    def massive_datasets() -> dict[str, list[dict]]:
        datasets = [asdict(item) for item in cache_store.list_datasets()]
        return {"datasets": datasets}

    @app.get("/operator/summary")
    def operator_summary() -> dict:
        status = live.status()
        latest = sqlite.list_trader_outputs(limit=20)
        warnings = sqlite.list_live_errors(limit=20)
        return {
            "live_status": status,
            "latest_signals": latest,
            "recent_warnings": warnings,
            "replay_runs": sqlite.list_replay_runs()[:10],
            "datasets": [asdict(item) for item in cache_store.list_datasets()[:10]],
            "core_symbols": CORE,
        }

    @app.get("/live/signals/{ticker}")
    def live_signal_for_ticker(ticker: str) -> dict:
        sig = live.latest_signals.get(ticker.upper())
        if sig is None:
            raise HTTPException(status_code=404, detail="ticker not found")
        return sig

    @app.get("/live/bars/{ticker}")
    def live_bars_for_ticker(ticker: str, limit: int = 200) -> list[dict]:
        rows = live.bars.get_bars(ticker, limit=limit)
        for row in rows:
            row["timestamp"] = row["timestamp"].isoformat()
        return rows

    @app.get("/live/events/{ticker}")
    def live_events_for_ticker(ticker: str, limit: int = 200) -> list[dict]:
        rows = live.buffer.get_events(ticker, limit=limit)
        for row in rows:
            row["timestamp"] = row["timestamp"].isoformat()
        return rows

    @app.get("/replay/runs")
    def replay_runs() -> list[dict]:
        return sqlite.list_replay_runs()

    @app.get("/replay/runs/{run_id}")
    def replay_run(run_id: str) -> dict:
        run = sqlite.get_replay_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/replay/runs/{run_id}/signals")
    def replay_signals(run_id: str) -> list[dict]:
        return sqlite.get_replay_signals(run_id)

    @app.get("/replay/runs/{run_id}/export")
    def replay_export(run_id: str) -> FileResponse:
        run = sqlite.get_replay_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        csv_path = Path(run.get("csv_path", ""))
        if not csv_path.exists():
            rows = sqlite.get_replay_signals(run_id)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(rows[0].keys()) if rows else ["timestamp", "ticker"]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        return FileResponse(csv_path)

    @app.get("/signals/{asset}")
    def by_asset(asset: str, limit: int = 100) -> list[dict]:
        return [sig.model_dump(mode="json") for sig in evidence.by_asset(asset.upper(), limit)]

    @app.get("/evidence/{signal_id}")
    def evidence_by_id(signal_id: str) -> dict:
        record = evidence.get_evidence(signal_id)
        if record is None:
            raise HTTPException(status_code=404, detail="signal not found")
        return record.model_dump(mode="json")

    return app


app = create_app()
