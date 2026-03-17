from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


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
    latest_result_exists: bool
    auth_configured: bool
    persistence_available: bool


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
        return HealthResponse(
            status="ok" if persistence_available else "degraded",
            version=app.version,
            latest_result_exists=service_instance.get_latest_result() is not None,
            auth_configured=bool(api_key),
            persistence_available=persistence_available,
        )

    @app.post("/ingest")
    def ingest(payload: IngestRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            return service_instance.ingest_payload(payload.model_dump(exclude_none=True))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/ingest/batch")
    def ingest_batch(payload: BatchIngestRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            results = service_instance.ingest_batch(
                [item.model_dump(exclude_none=True) for item in payload.items]
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"count": len(results), "results": results}

    @app.post("/ingest/csv")
    def ingest_csv(payload: CsvIngestRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            results = service_instance.ingest_csv(payload.csv_text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"count": len(results), "results": results}

    @app.post("/reset")
    def reset(_: None = Depends(require_api_key)) -> dict[str, bool]:
        service_instance.reset()
        return {"ok": True}

    @app.get("/results/latest")
    def get_latest() -> dict[str, Any]:
        latest = service_instance.get_latest_result()
        if latest is None:
            return {"result": None}
        return {"result": latest}

    @app.get("/results/recent")
    def get_recent(limit: int = 100) -> dict[str, Any]:
        results = service_instance.list_recent_results(limit=limit)
        return {"count": len(results), "results": results}

    return app


app = create_app()
