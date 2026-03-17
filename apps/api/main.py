from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


logger = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    timestamp: str | None = None
    site_id: str | None = None
    asset_id: str | None = None
    sensor_values: dict[str, Any] = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    items: list[IngestRequest]


class CsvIngestRequest(BaseModel):
    csv_text: str


class HealthResponse(BaseModel):
    status: str
    version: str
    auth_configured: bool
    persistence_available: bool
    latest_result_available: bool


class ResultsEnvelope(BaseModel):
    latest: dict[str, Any] | None = None
    count: int
    results: list[dict[str, Any]]


class ActionResponse(BaseModel):
    ok: bool


def _persistence_available(db_path: str) -> bool:
    try:
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with db_file.open("a", encoding="utf-8"):
            pass
        return True
    except OSError:
        return False


def is_api_key_valid(configured_key: str | None, provided_key: str | None) -> bool:
    if not configured_key:
        return True
    return configured_key == provided_key


def _results_envelope(results: list[dict[str, Any]], latest: dict[str, Any] | None) -> dict[str, Any]:
    return {"latest": latest, "count": len(results), "results": results}


def create_app(service: StructuralMonitoringService | None = None) -> FastAPI:
    api_key = os.getenv("NERAIUM_API_KEY")
    db_path = os.getenv("NERAIUM_DB_PATH", "neraium.db")

    app = FastAPI(title="Neraium SII API", version="0.1.0")
    persistence_available = _persistence_available(db_path)
    service_instance = service or StructuralMonitoringService(store=ResultStore(db_path=db_path))

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if not is_api_key_valid(api_key, x_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        latest = service_instance.get_latest_result()
        return HealthResponse(
            status="ok" if persistence_available else "degraded",
            version=app.version,
            auth_configured=bool(api_key),
            persistence_available=persistence_available,
            latest_result_available=latest is not None,
        )

    @app.post("/ingest", response_model=ResultsEnvelope)
    def ingest(payload: IngestRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        logger.info("ingest endpoint called")
        try:
            result = service_instance.ingest_payload(payload.model_dump(exclude_none=True))
        except ValueError as e:
            logger.warning("validation failure ingest: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        return _results_envelope([result], latest=result)

    @app.post("/ingest/batch", response_model=ResultsEnvelope)
    def ingest_batch(payload: BatchIngestRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        logger.info("ingest_batch endpoint called items=%s", len(payload.items))
        try:
            results = service_instance.ingest_batch(
                [item.model_dump(exclude_none=True) for item in payload.items]
            )
        except ValueError as e:
            logger.warning("validation failure ingest_batch: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        return _results_envelope(results, latest=results[-1] if results else None)

    @app.post("/ingest/csv", response_model=ResultsEnvelope)
    def ingest_csv(payload: CsvIngestRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        logger.info("ingest_csv endpoint called")
        try:
            results = service_instance.ingest_csv(payload.csv_text)
        except ValueError as e:
            logger.warning("validation failure ingest_csv: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        return _results_envelope(results, latest=results[-1] if results else None)

    @app.post("/reset", response_model=ActionResponse)
    def reset(_: None = Depends(require_api_key)) -> dict[str, bool]:
        logger.info("reset endpoint called")
        service_instance.reset()
        return {"ok": True}

    @app.get("/results/latest", response_model=ResultsEnvelope)
    def get_latest() -> dict[str, Any]:
        latest = service_instance.get_latest_result()
        results = [latest] if latest is not None else []
        return _results_envelope(results, latest=latest)

    @app.get("/results/recent", response_model=ResultsEnvelope)
    def get_recent(limit: int = 100) -> dict[str, Any]:
        results = service_instance.list_recent_results(limit=limit)
        latest = results[0] if results else None
        return _results_envelope(results, latest=latest)

    return app


app = create_app()
