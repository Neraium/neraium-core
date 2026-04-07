"""FastAPI endpoints for Neraium Markets."""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from neraium_core.markets.data.alignment import align_to_shared_clock
from neraium_core.markets.data.cache_store import CacheStore
from neraium_core.markets.data.historical_ingest import HistoricalIngestService
from neraium_core.markets.data.market_data_loader import MarketDataLoader
from neraium_core.markets.data.validation import validate_market_frame
from neraium_core.markets.defaults import DEFAULT_LIVE_SYMBOLS, DEFAULT_LIVE_TIMEFRAME
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.integrations.massive.config import MassiveConfigError, load_massive_config
from neraium_core.markets.integrations.massive.health import check_massive_health
from neraium_core.markets.live.live_runner import LiveSessionRunner
from neraium_core.markets.persistence.sqlite_store import MarketsSQLiteStore
from neraium_core.markets.replay import run_signal_replay
from neraium_core.markets.signals.signal_generator import generate_signal_for_asset, to_evidence
from neraium_core.markets.state.state_vector import CORE, build_state_vector

TIMEFRAMES = ["daily", "1h", "15m"]
LOGGER = logging.getLogger(__name__)


def _runtime_fingerprint() -> dict[str, str | None]:
    return {
        "app_identity": "neraium_core.markets.app.api:app",
        "version": "9.9.9-markets",
        "git_sha": os.getenv("NERAIUM_GIT_SHA") or os.getenv("GIT_SHA") or os.getenv("COMMIT_SHA"),
        "deploy_id": os.getenv("NERAIUM_DEPLOYMENT_ID") or os.getenv("APP_RUNNER_DEPLOYMENT_ID"),
        "service_id": os.getenv("APP_RUNNER_SERVICE_ID"),
        "service_url": os.getenv("APP_RUNNER_SERVICE_URL"),
    }


def _log_runtime_banner(data_dir: Path) -> None:
    fingerprint = _runtime_fingerprint()
    LOGGER.info("=" * 72)
    LOGGER.info("NERAIUM MARKETS API LIVE BOOT")
    LOGGER.info("app_identity=%s", fingerprint["app_identity"])
    LOGGER.info("version=%s", fingerprint["version"])
    LOGGER.info("git_sha=%s", fingerprint["git_sha"] or "unset")
    LOGGER.info("deploy_id=%s", fingerprint["deploy_id"] or "unset")
    LOGGER.info("service_id=%s", fingerprint["service_id"] or "unset")
    LOGGER.info("service_url=%s", fingerprint["service_url"] or "unset")
    LOGGER.info("data_dir=%s", data_dir)
    LOGGER.info("=" * 72)


class HistoricalFetchBody(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["SPY", "QQQ", "AAPL", "NVDA"])
    timeframe: str = "15m"
    start_date: str
    end_date: str


class LiveStartBody(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: DEFAULT_LIVE_SYMBOLS.copy())
    timeframe: str = DEFAULT_LIVE_TIMEFRAME


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
    app = FastAPI(title="NERAIUM MARKETS API LIVE", version="9.9.9-markets")
    evidence = EvidenceLog(path=evidence_path)
    cache_store = CacheStore()
    ingest = HistoricalIngestService(cache_store)
    sqlite = MarketsSQLiteStore()
    live = LiveSessionRunner()
    configured_data_dir = Path(data_dir)
    app.state.data_dir = configured_data_dir
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="markets-static")

    @app.on_event("startup")
    async def startup_banner() -> None:
        _log_runtime_banner(configured_data_dir)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"ok": True, "service": "markets", **_runtime_fingerprint()}

    @app.get("/", response_class=HTMLResponse)
    def root() -> HTMLResponse:
        return operator_ui()

    @app.get("/operator", response_class=HTMLResponse)
    def operator_ui() -> HTMLResponse:
        html_path = Path(__file__).parent / "static" / "operator.html"
        html = (
            html_path.read_text(encoding="utf-8")
            .replace("__DEFAULT_LIVE_SYMBOLS__", ",".join(DEFAULT_LIVE_SYMBOLS))
            .replace("__DEFAULT_LIVE_TIMEFRAME__", DEFAULT_LIVE_TIMEFRAME)
        )
        return HTMLResponse(html)

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
        health = check_massive_health(cfg)
        live_status = live.status()
        recent_error = sqlite.list_live_errors(limit=1)[0] if sqlite.list_live_errors(limit=1) else None
        payload = {
            **health.as_dict(),
            "provider": "massive",
            "latest_connectivity_check": datetime.now(timezone.utc).isoformat(),
            "recent_fetch_success": bool(sqlite.list_cached_datasets(limit=1)),
            "recent_live_event_at": live_status.get("last_event_at"),
            "recent_live_signal_at": live_status.get("last_signal_at"),
            "recent_error": recent_error,
            "session_state": live_status.get("session_state"),
            "readiness_state": live_status.get("readiness_state"),
        }
        sqlite.record_provider_health("massive", payload)
        return payload

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
    async def live_start(body: LiveStartBody | None = None) -> dict:
        start_body = body or LiveStartBody()
        effective_symbols = [item.upper() for item in (start_body.symbols or DEFAULT_LIVE_SYMBOLS)]
        effective_timeframe = start_body.timeframe or DEFAULT_LIVE_TIMEFRAME
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
        provider_health = check_massive_health(load_massive_config())
        if provider_health.status == "invalid_api_key":
            detail = provider_health.error or provider_health.status.replace("_", " ")
            raise HTTPException(status_code=503, detail=f"Massive provider unavailable: {detail}")
        try:
            await live.start(effective_symbols, effective_timeframe)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    def signal_history(
        limit: int = 300,
        ticker: str | None = None,
        session_type: str | None = None,
        action_permission: str | None = None,
        best_action: str | None = None,
        start_at: str | None = None,
        end_at: str | None = None,
        include_suppressed: bool = False,
    ) -> dict[str, list[dict] | dict]:
        ticker_upper = ticker.upper() if ticker else None
        rows: list[dict] = []
        include_live = session_type in (None, "live")
        include_replay = session_type in (None, "replay")
        if include_live:
            for row in sqlite.list_trader_outputs(
                limit=limit,
                ticker=ticker_upper,
                action_permission=action_permission,
                best_action=best_action,
                start_at=start_at,
                end_at=end_at,
            ):
                rows.append({**row, "session_type": "live", "emitted": True})
            if include_suppressed:
                for row in sqlite.list_suppressed_outputs(
                    limit=limit,
                    ticker=ticker_upper,
                    action_permission=action_permission,
                    best_action=best_action,
                    start_at=start_at,
                    end_at=end_at,
                ):
                    rows.append({**row, "session_type": "live", "emitted": False})
        if include_replay:
            for row in sqlite.list_replay_outputs(
                limit=limit,
                ticker=ticker_upper,
                action_permission=action_permission,
                best_action=best_action,
                start_at=start_at,
                end_at=end_at,
            ):
                rows.append({**row, "session_type": "replay", "emitted": True})
        rows.sort(key=lambda item: item.get("timestamp") or item.get("created_at") or "", reverse=True)
        summary = {
            "suppression_count": sum(1 for row in rows if row.get("emitted") is False),
            "abstention_count": sum(1 for row in rows if row.get("action_permission") == "ABSTAIN"),
        }
        return {"signals": rows[:limit], "summary": summary}

    @app.get("/integrations/massive/datasets")
    def massive_datasets() -> dict[str, list[dict]]:
        datasets = [asdict(item) for item in cache_store.list_datasets()]
        return {"datasets": datasets}

    @app.get("/operator/summary")
    def operator_summary() -> dict:
        status = live.status()
        latest = sqlite.list_trader_outputs(limit=20)
        warnings = sqlite.list_live_errors(limit=20)
        provider_cfg = load_massive_config()
        provider_health = check_massive_health(provider_cfg)
        checklist = [
            {"key": "provider_configured", "ok": provider_health.config_present and provider_health.api_key_present},
            {"key": "provider_reachable", "ok": provider_health.status not in {"invalid_api_key", "rest_unreachable", "missing_api_key"}},
            {"key": "symbols_valid", "ok": bool(status["symbols"])},
            {"key": "core_symbols_present", "ok": all(sym in set(status["symbols"]) for sym in CORE)},
            {"key": "timeframe_selected", "ok": status["timeframe"] in {"1m", "5m", "15m"}},
            {"key": "incoming_data_received", "ok": bool(status.get("last_event_at"))},
            {"key": "warmup_complete", "ok": status["warmup_progress"] >= 1.0},
            {"key": "signal_engine_active", "ok": bool(status.get("last_signal_at"))},
        ]
        return {
            "live_status": status,
            "latest_signals": latest,
            "recent_warnings": warnings,
            "replay_runs": sqlite.list_replay_runs()[:10],
            "datasets": [asdict(item) for item in cache_store.list_datasets()[:10]],
            "core_symbols": CORE,
            "suppressed_recent": sqlite.list_suppressed_outputs(limit=20),
            "launch_checklist": checklist,
            "defaults": {"symbols": DEFAULT_LIVE_SYMBOLS, "timeframe": DEFAULT_LIVE_TIMEFRAME},
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


def create_app_safe() -> FastAPI:
    try:
        print("STARTING NERAIUM MARKETS API LIVE")
        return create_app()
    except Exception as exc:
        import traceback

        print("STARTUP FAILURE:", repr(exc))
        traceback.print_exc()

        fallback = FastAPI(title="NERAIUM MARKETS API FALLBACK", version="9.9.9-fallback")

        @fallback.get("/")
        def root() -> dict[str, str]:
            return {"status": "degraded", "app": "markets-fallback", "error": str(exc)}

        @fallback.get("/health")
        def health() -> dict[str, object]:
            return {"ok": False, "service": "markets-fallback", "error": str(exc)}

        return fallback


app = create_app_safe()
