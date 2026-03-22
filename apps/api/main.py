from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from apps.api.web import build_web_router
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


class CreateRunRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    activate: bool = True
    config: dict[str, Any] = Field(default_factory=dict)


class UpdateRunRequest(BaseModel):
    name: str | None = None
    config: dict[str, Any] | None = None
    status: str | None = None


class ActivateRunRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=200)


class ExportEnvelope(BaseModel):
    run_id: str | None = None
    format: Literal["json", "csv"]
    count: int
    content_type: str
    filename: str
    content: str


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


class RunEnvelope(BaseModel):
    run: dict[str, Any] | None


class RunsEnvelope(BaseModel):
    active_run: dict[str, Any] | None = None
    count: int
    runs: list[dict[str, Any]]


class ResultEnvelope(BaseModel):
    result: dict[str, Any]


def _ensure_default_run(service: StructuralMonitoringService) -> dict[str, Any]:
    existing = service.get_active_run()
    if existing is not None:
        return existing
    return service.create_run(
        name="Default Run",
        config={"source": "api-default"},
        activate=True,
    )
 

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


def _resolve_run_id(service: StructuralMonitoringService, run_id: str | None) -> str | None:
    if run_id is not None and str(run_id).strip():
        return str(run_id).strip()
    active = service.get_active_run()
    if active is None:
        return None
    rid = active.get("run_id")
    if rid is None:
        return None
    return str(rid)


def _require_run_id(service: StructuralMonitoringService, run_id: str | None) -> str:
    resolved = _resolve_run_id(service, run_id)
    if resolved is None:
        raise HTTPException(status_code=400, detail="No active run. Create or activate a run first.")
    return resolved


def _request_run_id_or_active(service: StructuralMonitoringService, run_id: str | None) -> str | None:
    if run_id is None:
        return _resolve_run_id(service, None)
    text = str(run_id).strip()
    if not text:
        return _resolve_run_id(service, None)
    return text


def _resolve_run_id_with_default(service: StructuralMonitoringService, run_id: str | None) -> str:
    resolved = _request_run_id_or_active(service, run_id)
    if resolved is not None:
        return resolved
    created = _ensure_default_run(service)
    return str(created.get("run_id"))


def _csv_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    if any(ch in text for ch in [",", "\"", "\n", "\r"]):
        return "\"" + text.replace("\"", "\"\"") + "\""
    return text


def _build_export(results: list[dict[str, Any]], *, format_name: Literal["json", "csv"]) -> tuple[str, str]:
    if format_name == "json":
        return ("application/json; charset=utf-8", json.dumps(results, indent=2, sort_keys=False))

    header = [
        "result_id",
        "run_id",
        "timestamp",
        "site_id",
        "asset_id",
        "state",
        "phase",
        "risk_level",
        "trend",
        "operator_message",
        "structural_drift_score",
        "composite_instability",
        "persisted_at",
    ]
    lines = [",".join(header)]
    for row in results:
        composite = None
        analytics = row.get("experimental_analytics")
        if isinstance(analytics, dict):
            raw = analytics.get("composite_instability")
            if raw is not None:
                try:
                    composite = float(raw)
                except (TypeError, ValueError):
                    composite = None
        data = [
            row.get("result_id"),
            row.get("run_id"),
            row.get("timestamp"),
            row.get("site_id"),
            row.get("asset_id"),
            row.get("state"),
            row.get("phase"),
            row.get("risk_level"),
            row.get("trend"),
            row.get("operator_message"),
            row.get("structural_drift_score"),
            composite,
            row.get("persisted_at"),
        ]
        lines.append(",".join(_csv_escape(v) for v in data))
    return ("text/csv; charset=utf-8", "\n".join(lines))


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

    @app.post("/runs", response_model=RunEnvelope)
    def create_run(payload: CreateRunRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            run = service_instance.create_run(
                name=payload.name.strip(),
                config=dict(payload.config),
                activate=bool(payload.activate),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"run": run}

    @app.post("/runs/activate", response_model=RunEnvelope)
    def activate_run(payload: ActivateRunRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            run = service_instance.activate_run(payload.run_id.strip())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run": run}

    @app.get("/runs", response_model=RunsEnvelope)
    def list_runs(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        runs = service_instance.list_runs(limit=limit)
        active = service_instance.get_active_run()
        return {"active_run": active, "count": len(runs), "runs": runs}

    @app.get("/runs/active", response_model=RunEnvelope)
    def get_active_run() -> dict[str, Any]:
        run = service_instance.get_active_run()
        if run is None:
            return {"run": None}
        return {"run": run}

    @app.get("/runs/{run_id}", response_model=RunEnvelope)
    def get_run(run_id: str) -> dict[str, Any]:
        run = service_instance.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return {"run": run}

    @app.patch("/runs/{run_id}", response_model=RunEnvelope)
    def update_run(run_id: str, payload: UpdateRunRequest, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            run = service_instance.update_run(
                run_id,
                name=payload.name,
                config=payload.config,
                status=payload.status,
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if "Unknown run_id" in detail else 400
            raise HTTPException(status_code=status_code, detail=detail)
        return {"run": run}

    @app.post("/runs/{run_id}/activate", response_model=RunEnvelope)
    def activate_run_path(run_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
        try:
            run = service_instance.activate_run(run_id.strip())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run": run}

    @app.post("/ingest", response_model=ResultsEnvelope)
    def ingest(
        payload: IngestRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        logger.info("ingest endpoint called")
        try:
            resolved = _resolve_run_id_with_default(service_instance, run_id)
            result = service_instance.ingest_payload(
                payload.model_dump(exclude_none=True),
                run_id=resolved,
            )
        except ValueError as e:
            logger.warning("validation failure ingest: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        return _results_envelope([result], latest=result)

    @app.post("/ingest/batch", response_model=ResultsEnvelope)
    def ingest_batch(
        payload: BatchIngestRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        logger.info("ingest_batch endpoint called items=%s", len(payload.items))
        try:
            resolved = _resolve_run_id_with_default(service_instance, run_id)
            results = service_instance.ingest_batch(
                [item.model_dump(exclude_none=True) for item in payload.items],
                run_id=resolved,
            )
        except ValueError as e:
            logger.warning("validation failure ingest_batch: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        return _results_envelope(results, latest=results[-1] if results else None)

    @app.post("/ingest/csv", response_model=ResultsEnvelope)
    def ingest_csv(
        payload: CsvIngestRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        logger.info("ingest_csv endpoint called")
        try:
            resolved = _resolve_run_id_with_default(service_instance, run_id)
            results = service_instance.ingest_csv(payload.csv_text, run_id=resolved)
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
    def get_latest(run_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved = _resolve_run_id(service_instance, run_id)
        latest = service_instance.get_latest_result(run_id=resolved)
        results = [latest] if latest is not None else []
        return _results_envelope(results, latest=latest)

    @app.get("/results/recent", response_model=ResultsEnvelope)
    def get_recent(limit: int = 100, run_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved = _resolve_run_id(service_instance, run_id)
        results = service_instance.list_recent_results(limit=limit, run_id=resolved)
        latest = results[0] if results else None
        return _results_envelope(results, latest=latest)

    @app.get("/results/export", response_model=ExportEnvelope)
    def export_results(
        format: Literal["json", "csv"] = Query(default="json"),
        limit: int = Query(default=500, ge=1, le=5000),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved = _resolve_run_id(service_instance, run_id)
        results = service_instance.list_recent_results(limit=limit, run_id=resolved)
        content_type, content = _build_export(results, format_name=format)
        suffix = "json" if format == "json" else "csv"
        file_id = resolved or "all_runs"
        filename = f"neraium_results_{file_id}.{suffix}"
        return {
            "run_id": resolved,
            "format": format,
            "count": len(results),
            "content_type": content_type,
            "filename": filename,
            "content": content,
        }

    @app.get("/results/{result_id}", response_model=ResultEnvelope)
    def get_result_by_id(result_id: int, run_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved = _resolve_run_id(service_instance, run_id)
        result = service_instance.get_result_by_id(result_id, run_id=resolved)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        return {"result": result}

    @app.get("/export", response_model=ExportEnvelope)
    def export_results_legacy(
        format: Literal["json", "csv"] = Query(default="json"),
        limit: int = Query(default=500, ge=1, le=5000),
        run_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return export_results(format=format, limit=limit, run_id=run_id)

    app.include_router(build_web_router())

    return app


app = create_app()
