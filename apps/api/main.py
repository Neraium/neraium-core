from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from apps.api.web import build_web_router
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


logger = logging.getLogger(__name__)

DEFAULT_MAX_REQUEST_BODY_BYTES = 50 * 1024 * 1024
DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE = DEFAULT_MAX_REQUEST_BODY_BYTES


class RequestBodyTooLargeError(Exception):
    """Raised when an incoming request body exceeds configured max size."""


class MaxRequestBodySizeMiddleware:
    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max(1, int(max_body_size))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        raw_content_length = headers.get(b"content-length")
        if raw_content_length:
            try:
                content_length = int(raw_content_length.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                content_length = None
            if content_length is not None and content_length > self.max_body_size:
                await self._send_413(scope, receive, send)
                return

        bytes_seen = 0
        response_started = False

        async def guarded_receive() -> Message:
            nonlocal bytes_seen
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                bytes_seen += len(body)
                if bytes_seen > self.max_body_size:
                    raise RequestBodyTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, tracked_send)
        except RequestBodyTooLargeError:
            if not response_started:
                await self._send_413(scope, receive, send)

    async def _send_413(self, scope: Scope, receive: Receive, send: Send) -> None:
        max_mb = self.max_body_size / (1024 * 1024)
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": f"Request body too large (max {max_mb:.1f}MB)."},
        )
        await response(scope, receive, send)


def _request_body_limit_bytes() -> int:
    raw = os.getenv("NERAIUM_MAX_REQUEST_BODY_BYTES")
    if not raw:
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid NERAIUM_MAX_REQUEST_BODY_BYTES=%r; using default=%s",
            raw,
            DEFAULT_MAX_REQUEST_BODY_BYTES,
        )
        return DEFAULT_MAX_REQUEST_BODY_BYTES
    return max(value, DEFAULT_MAX_REQUEST_BODY_BYTES)


def _uvicorn_h11_max_incomplete_event_size() -> int:
    raw = os.getenv("NERAIUM_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE")
    if not raw:
        return DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid NERAIUM_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE=%r; using default=%s",
            raw,
            DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE,
        )
        return DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE
    return max(value, DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE)


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


class GeometryEnvelope(BaseModel):
    run_id: str | None = None
    result_id: int | None = None
    timestamp: str | None = None
    available: bool
    reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    projection: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return float(f)


def _to_square_matrix(value: Any) -> np.ndarray | None:
    try:
        mat = np.asarray(value, dtype=float)
    except Exception:
        return None
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        return None
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    # Correlation geometry is expected to be self-correlated on diagonal.
    np.fill_diagonal(mat, 1.0)
    return mat


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-9:
        return np.full_like(arr, 0.5, dtype=float)
    return (arr - lo) / (hi - lo)


def _project_geometry_positions(
    corr_current: np.ndarray,
    *,
    node_stress: np.ndarray,
    corr_baseline: np.ndarray | None,
) -> np.ndarray:
    n = int(corr_current.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if n == 1:
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float)

    vals, vecs = np.linalg.eigh(corr_current)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    def _axis(component_idx: int) -> np.ndarray:
        if component_idx >= vecs.shape[1]:
            return np.zeros((n,), dtype=float)
        scale = float(np.sqrt(max(vals[component_idx], 1e-6)))
        return np.asarray(vecs[:, component_idx], dtype=float) * scale

    axis_x = _axis(0)
    axis_y = _axis(1)
    if corr_baseline is not None and corr_baseline.shape == corr_current.shape:
        axis_z = np.mean(np.abs(corr_current - corr_baseline), axis=1)
    else:
        axis_z = _axis(2)

    for axis in (axis_x, axis_y, axis_z):
        max_abs = float(np.max(np.abs(axis))) if axis.size else 0.0
        if max_abs > 1e-9:
            axis /= max_abs

    # Fallback to a deterministic ring if matrix eigenvectors are near-degenerate.
    spread = float(np.std(axis_x) + np.std(axis_y))
    if spread < 1e-5:
        theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
        axis_x = np.cos(theta)
        axis_y = np.sin(theta)

    axis_z = 0.55 * axis_z + 0.45 * (2.0 * _normalize_vector(node_stress) - 1.0)
    max_abs_z = float(np.max(np.abs(axis_z))) if axis_z.size else 0.0
    if max_abs_z > 1e-9:
        axis_z /= max_abs_z

    return np.stack([axis_x, axis_y, axis_z], axis=1)


def _build_geometry_payload(result: dict[str, Any], *, run_id: str | None) -> dict[str, Any]:
    analytics = result.get("experimental_analytics")
    analytics_dict = analytics if isinstance(analytics, dict) else {}
    corr_geometry = analytics_dict.get("correlation_geometry")
    corr_geometry_dict = corr_geometry if isinstance(corr_geometry, dict) else {}
    corr_current = _to_square_matrix(corr_geometry_dict.get("current"))
    corr_baseline = _to_square_matrix(corr_geometry_dict.get("baseline"))
    if corr_current is not None and corr_baseline is not None and corr_baseline.shape != corr_current.shape:
        corr_baseline = None

    feature_names: list[str] = []
    names_from_analytics = analytics_dict.get("valid_sensor_names") or analytics_dict.get("feature_names")
    if isinstance(names_from_analytics, list):
        feature_names = [str(v) for v in names_from_analytics if str(v).strip()]
    if not feature_names:
        raw = result.get("sensor_relationships")
        if isinstance(raw, list):
            feature_names = [str(v) for v in raw if str(v).strip()]

    metrics = {
        "state": result.get("state") or result.get("interpreted_state"),
        "phase": result.get("phase") or result.get("interpreted_state") or result.get("state"),
        "risk_level": result.get("risk_level", "UNKNOWN"),
        "trend": result.get("trend", "UNKNOWN"),
        "structural_drift_score": _safe_float(result.get("structural_drift_score"), 0.0),
        "composite_instability": _safe_float(
            result.get("latest_instability"),
            _safe_float(analytics_dict.get("composite_instability"), 0.0),
        ),
        "confidence": _safe_float(result.get("confidence"), _safe_float(result.get("confidence_score"), 0.0)),
    }
    projection = {
        "method": "spectral_projection_from_engine_correlation_geometry",
        "is_visualization_projection": True,
        "source": "engine correlation geometry + graph analytics",
        "note": (
            "Node positions are a deterministic visualization projection derived from engine "
            "correlation outputs; they are not the core SII computation space."
        ),
    }
    provenance = {
        "engine_fields": [
            "sensor_relationships",
            "experimental_analytics.correlation_geometry.current",
            "experimental_analytics.correlation_geometry.baseline",
            "experimental_analytics.signal_structural_importance",
            "risk_level",
            "state",
            "interpreted_state",
            "trend",
            "structural_drift_score",
            "latest_instability",
            "confidence/confidence_score",
        ],
        "positions": "deterministic projection from engine outputs",
    }

    try:
        result_id = int(result.get("result_id")) if result.get("result_id") is not None else None
    except (TypeError, ValueError):
        result_id = None

    if corr_current is None:
        return {
            "run_id": run_id or result.get("run_id"),
            "result_id": result_id,
            "timestamp": result.get("timestamp") or result.get("persisted_at"),
            "available": False,
            "reason": "Correlation geometry unavailable for this result.",
            "metrics": metrics,
            "nodes": [],
            "edges": [],
            "projection": projection,
            "provenance": provenance,
        }

    n = int(corr_current.shape[0])
    if len(feature_names) < n:
        feature_names = feature_names + [f"signal_{i + 1}" for i in range(len(feature_names), n)]
    if len(feature_names) > n:
        feature_names = feature_names[:n]

    importance_raw = analytics_dict.get("signal_structural_importance")
    if isinstance(importance_raw, list) and len(importance_raw) >= n:
        importance = np.asarray([_safe_float(v, 0.0) for v in importance_raw[:n]], dtype=float)
    else:
        corr_abs = np.abs(corr_current - np.eye(n))
        importance = np.mean(corr_abs, axis=1)
    importance_norm = _normalize_vector(importance)

    if corr_baseline is not None and corr_baseline.shape == corr_current.shape:
        stress_raw = np.mean(np.abs(corr_current - corr_baseline), axis=1)
    else:
        stress_raw = importance.copy()
    stress_norm = _normalize_vector(stress_raw)
    positions = _project_geometry_positions(corr_current, node_stress=stress_norm, corr_baseline=corr_baseline)

    nodes: list[dict[str, Any]] = []
    for idx in range(n):
        stress = float(stress_norm[idx])
        state = "stable"
        if stress >= 0.66:
            state = "critical"
        elif stress >= 0.33:
            state = "watch"
        nodes.append(
            {
                "id": str(feature_names[idx]),
                "label": str(feature_names[idx]),
                "position": {
                    "x": round(float(positions[idx, 0]), 6),
                    "y": round(float(positions[idx, 1]), 6),
                    "z": round(float(positions[idx, 2]), 6),
                },
                "magnitude": round(float(importance_norm[idx]), 6),
                "stress": round(stress, 6),
                "state": state,
                "role": "signal",
            }
        )

    edges: list[dict[str, Any]] = []
    if n > 1:
        upper_idx = np.triu_indices(n, k=1)
        upper_abs = np.abs(corr_current[upper_idx])
        if upper_abs.size:
            threshold = float(np.clip(np.percentile(upper_abs, 72.0), 0.22, 0.78))
        else:
            threshold = 1.1
        for i in range(n):
            for j in range(i + 1, n):
                weight = float(corr_current[i, j])
                magnitude = abs(weight)
                if magnitude < threshold:
                    continue
                baseline_weight = float(corr_baseline[i, j]) if corr_baseline is not None else 0.0
                delta = weight - baseline_weight
                edges.append(
                    {
                        "source": str(feature_names[i]),
                        "target": str(feature_names[j]),
                        "weight": round(weight, 6),
                        "magnitude": round(magnitude, 6),
                        "delta": round(delta, 6),
                        "type": "positive" if weight >= 0.0 else "negative",
                    }
                )
        edges.sort(key=lambda e: float(e.get("magnitude", 0.0)), reverse=True)
        edges = edges[:240]

    return {
        "run_id": run_id or result.get("run_id"),
        "result_id": result_id,
        "timestamp": result.get("timestamp") or result.get("persisted_at"),
        "available": True,
        "reason": None,
        "metrics": metrics,
        "nodes": nodes,
        "edges": edges,
        "projection": projection,
        "provenance": provenance,
    }


def create_app(
    service: StructuralMonitoringService | None = None,
    *,
    max_request_body_bytes: int | None = None,
) -> FastAPI:
    api_key = os.getenv("NERAIUM_API_KEY")
    db_path = os.getenv("NERAIUM_DB_PATH", "neraium.db")
    request_body_limit = (
        int(max_request_body_bytes)
        if max_request_body_bytes is not None
        else _request_body_limit_bytes()
    )

    app = FastAPI(title="Neraium SII API", version="0.1.0")
    app.add_middleware(MaxRequestBodySizeMiddleware, max_body_size=request_body_limit)
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

    @app.get("/runs/{run_id}/geometry", response_model=GeometryEnvelope)
    def get_run_geometry(
        run_id: str,
        result_id: int | None = Query(default=None),
    ) -> dict[str, Any]:
        run = service_instance.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

        if result_id is not None:
            result = service_instance.get_result_by_id(result_id, run_id=run_id)
            if result is None:
                raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        else:
            result = service_instance.get_latest_result(run_id=run_id)

        if result is None:
            return {
                "run_id": run_id,
                "result_id": None,
                "timestamp": None,
                "available": False,
                "reason": "No results available for this run yet.",
                "metrics": {},
                "nodes": [],
                "edges": [],
                "projection": {
                    "method": "spectral_projection_from_engine_correlation_geometry",
                    "is_visualization_projection": True,
                    "source": "engine correlation geometry + graph analytics",
                    "note": (
                        "Node positions are a deterministic visualization projection derived from engine "
                        "correlation outputs; they are not the core SII computation space."
                    ),
                },
                "provenance": {
                    "engine_fields": [
                        "sensor_relationships",
                        "experimental_analytics.correlation_geometry.current",
                        "experimental_analytics.correlation_geometry.baseline",
                    ],
                    "positions": "deterministic projection from engine outputs",
                },
            }

        return _build_geometry_payload(result, run_id=run_id)

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

    @app.get("/results/{result_id}/geometry", response_model=GeometryEnvelope)
    def get_result_geometry(result_id: int, run_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved = _resolve_run_id(service_instance, run_id)
        result = service_instance.get_result_by_id(result_id, run_id=resolved)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        return _build_geometry_payload(result, run_id=result.get("run_id") or resolved)

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        h11_max_incomplete_event_size=_uvicorn_h11_max_incomplete_event_size(),
    )
