"""FastAPI endpoints for Neraium Markets."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from neraium_core.markets.data.alignment import align_to_shared_clock
from neraium_core.markets.data.market_data_loader import MarketDataLoader
from neraium_core.markets.data.validation import validate_market_frame
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.signals.signal_generator import generate_signal_for_asset, to_evidence
from neraium_core.markets.state.state_vector import CORE, build_state_vector

TIMEFRAMES = ["daily", "1h", "15m"]


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
    app = FastAPI(title="Neraium Markets API", version="0.1.0")
    evidence = EvidenceLog(path=evidence_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "neraium-markets"}

    @app.post("/run-signals")
    def run_signals() -> dict[str, list[dict]]:
        signals = run_signal_pipeline(Path(data_dir), evidence)
        return {"signals": signals}

    @app.get("/signals/latest")
    def latest_signals(limit: int = 20) -> list[dict]:
        return [sig.model_dump(mode="json") for sig in evidence.latest(limit)]

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
