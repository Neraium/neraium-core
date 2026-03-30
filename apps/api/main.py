from __future__ import annotations

import base64
import json
import logging
import mimetypes
import math
import os
import tempfile
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .integration import (
    IntegrationMappingError,
    apply_integration_mapping,
    load_integration_config,
    resolve_customer_integration,
)
from .web import build_web_router
from .routers.health import build_health_router
from .routers.alerts import build_alerts_router
from .routers.geometry import build_geometry_router
from .services.alerts import alert_thresholds as service_alert_thresholds, evaluate_alerts, dispatch_alert_stubs
from .services.request_context import (
    resolve_customer_id,
    resolve_run_id,
    require_run_id,
    request_run_id_or_active,
    resolve_run_id_with_default,
)
from .services.validation_utils import actionable_validation_detail
from .services.export_utils import build_export
from .services.geometry import (
    STRUCTURAL_FLOW_PLANE_MAX_N,
    _build_geometry_edges,
    _diamond_plane_positions_four,
    _plane_ring_positions,
)
from ._core_imports import (
    ResultStore,
    StructuralMonitoringService,
    log_structured,
    infer_semantic_mapping,
    parse_csv_sample_for_mapping,
    resolve_mapping,
    row_to_frame_kwargs,
    validate_mapping,
    summarize_exception_for_logs,
)
from neraium_core.ingestion_normalization import normalize_external_batch_payload, normalize_external_payload


logger = logging.getLogger(__name__)

# Windows / older Python may omit .mjs; browsers refuse module scripts with wrong MIME.
mimetypes.add_type("text/javascript", ".mjs", strict=False)


class CacheControlStaticFiles(StaticFiles):
    """Static files with cache headers tuned for cloud delivery.

    HTML stays non-cacheable to allow clean deploy updates.
    Versionable assets (js/css/images/fonts) get long-lived public caching.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.setdefault("Vary", "Accept-Encoding")
        ext = Path(path).suffix.lower()
        query_string = scope.get("query_string", b"")
        query_params = parse_qs(query_string.decode("utf-8", errors="ignore")) if query_string else {}
        has_asset_version = bool(query_params.get("v", [None])[0])
        if ext in {".html"}:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif ext in {".js", ".mjs", ".css"}:
            if has_asset_version:
                response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
            else:
                response.headers["Cache-Control"] = "public, max-age=300, must-revalidate"
        elif ext in {".csv", ".txt", ".json", ".map"}:
            response.headers["Cache-Control"] = "public, max-age=3600, stale-while-revalidate=300"
        else:
            response.headers["Cache-Control"] = "public, max-age=604800, stale-while-revalidate=86400"
        return response


def _mount_web_static(app: FastAPI) -> None:
    """Serve `apps/api/static` at `/web` (app.js, styles, three-init, …).

    Uses Path(__file__) so the directory is correct regardless of process cwd.
    Registered after the web router so explicit HTML routes win; /web/* is fully static.

    If this mount is skipped, GET /web/... falls through to FastAPI's default 404
    (JSON ``{"detail":"Not Found"}``), which is easy to mistake for an API error.
    """
    static_dir = Path(__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        logger.error(
            "Web static directory missing: %s — /web/* will 404. Clone or sync apps/api/static.",
            static_dir,
        )
        return
    app.mount(
        "/web",
        CacheControlStaticFiles(directory=str(static_dir)),
        name="web",
    )
    logger.info("Serving static files at /web from %s", static_dir)
    # Front-end modules are loaded via CDN import map in static/index.html.
    # Keep startup resilient when /web/vendor/three is not deployed.


DEFAULT_MAX_REQUEST_BODY_BYTES = 50 * 1024 * 1024
# Keep parser allowance above app-level request cap so oversize requests
# are handled by middleware with a clean 413 response instead of reset.
DEFAULT_UVICORN_H11_MAX_INCOMPLETE_EVENT_SIZE = 64 * 1024 * 1024
DEFAULT_UPLOAD_STREAM_CHUNK_BYTES = 1024 * 1024
DEFAULT_INGEST_JOB_MAX_ERROR_SAMPLES = 25
DEFAULT_CORS_ALLOW_ORIGINS: tuple[str, ...] = ()
DEFAULT_CORS_ALLOW_HEADERS = (
    "Content-Type",
    "Authorization",
    "X-API-Key",
    "Accept",
    # Browser tracing stacks (Sentry/OpenTelemetry) can attach these automatically.
    "baggage",
    "sentry-trace",
    "traceparent",
    "tracestate",
)


def _configure_logging() -> None:
    raw = str(os.getenv("NERAIUM_LOG_LEVEL", "INFO")).strip().upper()
    level = getattr(logging, raw, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


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


def _cors_allow_origins() -> list[str]:
    raw = str(os.getenv("NERAIUM_CORS_ALLOW_ORIGINS") or "").strip()
    configured = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
    merged: list[str] = []
    for origin in [*DEFAULT_CORS_ALLOW_ORIGINS, *configured]:
        if origin and origin not in merged:
            merged.append(origin)
    return merged


def _cors_allow_headers() -> list[str]:
    raw = str(os.getenv("NERAIUM_CORS_ALLOW_HEADERS") or "").strip()
    configured = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
    merged: list[str] = []
    seen: set[str] = set()
    for header in [*DEFAULT_CORS_ALLOW_HEADERS, *configured]:
        normalized = header.lower() if header else ""
        if header and normalized not in seen:
            merged.append(header)
            seen.add(normalized)
    return merged


def _cors_allow_origin_regex() -> str | None:
    raw = str(os.getenv("NERAIUM_CORS_ALLOW_ORIGIN_REGEX") or "").strip()
    return raw or None


class IngestRequest(BaseModel):
    model_config = {"extra": "allow"}

    customer_id: str | None = None
    timestamp: str | None = None
    site_id: str | None = None
    asset_id: str | None = None
    sensor_values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = normalize_external_payload(data, customer_id=data.get("customer_id"))
        return {
            "customer_id": normalized.get("customer_id"),
            "timestamp": normalized.get("timestamp"),
            "site_id": normalized.get("site_id"),
            "asset_id": normalized.get("asset_id"),
            "sensor_values": normalized.get("sensor_values", {}),
        }


class IngestFrameRequest(BaseModel):
    """Production API payload for a single telemetry frame."""

    timestamp: str
    site_id: str
    asset_id: str
    sensor_values: dict[str, Any] = Field(default_factory=dict)
    customer_id: str | None = None


DEMO_STRUCTURAL_SENSOR_KEYS = [
    "pressure",
    "flow",
    "vibration",
    "temperature",
    "motor_current",
    "bearing_temp",
    "load_cell",
    "rpm",
    "humidity",
    "displacement",
    "valve_position",
    "shaft_accel",
    "lubrication_psi",
    "seismic_x",
    "seismic_y",
    "winding_temp",
    "inlet_guide",
    "outlet_guide",
    "torque_est",
    "casing_vibe",
    "oil_quality",
    "stator_temp",
    "field_bus_ok",
    "coolant_flow",
]


class DemoSeedRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    profile: Literal["sample", "stable", "watch", "critical"] = "sample"
    minutes: int = Field(default=120, ge=10, le=240)
    site_id: str = "demo-site"
    asset_id: str = "demo-asset"


class DemoCmapssStartRequest(BaseModel):
    customer_id: str | None = None
    max_frames: int = Field(default=10, ge=5, le=120)


def _build_demo_sensor_values_row(i: int, p: float, drift_lift: float, vib_spike: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, key in enumerate(DEMO_STRUCTURAL_SENSOR_KEYS):
        phase = k * 0.85
        wave = math.sin(i / (5.2 + k * 0.11) + phase)
        w2 = math.cos(i / (7.1 + k * 0.09) + phase * 0.65)
        base = 18 + k * 6.2
        out[key] = (
            base
            + wave * (1 + drift_lift * (0.45 + k * 0.025))
            + w2 * (0.55 + vib_spike * 0.12)
            + i * (0.011 + k * 0.0008)
            + p * (0.15 + k * 0.02)
        )
    return out


class BatchIngestRequest(BaseModel):
    items: list[IngestRequest]

    @model_validator(mode="before")
    @classmethod
    def _accept_records_alias(cls, data: Any) -> Any:
        """Accept legacy/front-end payloads that send `records` instead of `items`."""
        if isinstance(data, dict):
            if "items" in data:
                return data
            if any(k in data for k in ("records", "payloads", "stream")):
                normalized, _ = normalize_external_batch_payload(
                    data,
                    customer_id=data.get("customer_id"),
                )
                remapped = dict(data)
                remapped["items"] = normalized
                return remapped
        return data


class CsvColumnMappingPayload(BaseModel):
    """Semantic roles: which CSV columns map to time, entity, optional site, and numeric sensors."""

    timestamp: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    site_id: str | None = None
    sensor_columns: list[str] = Field(min_length=1)


class CsvIngestRequest(BaseModel):
    customer_id: str | None = None
    csv_text: str
    column_mapping: CsvColumnMappingPayload | None = None


class CsvPreviewRequest(BaseModel):
    csv_sample: str = Field(..., max_length=524_288)


class CsvPreviewResponse(BaseModel):
    headers: list[str]
    suggested_mapping: dict[str, Any] | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False


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


class LockBaselineRequest(BaseModel):
    locked: bool = True


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
    core_runtime_mode: str = "full"
    core_runtime_fallback: bool = False
    core_runtime_notes: list[str] = Field(default_factory=list)
    runtime_state_diagnostics: dict[str, Any] = Field(default_factory=dict)


class ClientErrorReport(BaseModel):
    """Browser-reported script errors (product UI telemetry)."""

    message: str = ""
    stack: str | None = None
    url: str | None = None
    source: str | None = None
    lineno: int | None = None
    colno: int | None = None
    reason: str | None = None


class ResultsEnvelope(BaseModel):
    status: str | None = None
    run_id: str | None = None
    processed: int | None = None
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
    views: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    projection: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    graph_analytics: dict[str, Any] | None = None
    system_state: dict[str, Any] | None = None


class IngestJobEnvelope(BaseModel):
    job_id: str
    status: str
    run_id: str | None = None
    customer_id: str
    filename: str
    created_at: str
    updated_at: str
    rows_processed: int = 0
    rows_succeeded: int = 0
    rows_failed: int = 0
    partial_success: bool = False
    upload_bytes_received: int = 0
    upload_bytes_total: int | None = None
    error_samples: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None
    latest_result: dict[str, Any] | None = None


class PullIntegrationStartRequest(BaseModel):
    endpoint_url: str | None = Field(default=None, min_length=1, max_length=2000)
    polling_interval_seconds: float | None = Field(default=None, ge=0.2, le=3600.0)
    auth_type: Literal["none", "basic", "bearer"] | None = None
    username: str | None = None
    password: str | None = None
    token: str | None = None
    run_id: str | None = None
    retry_max_attempts: int | None = Field(default=None, ge=1, le=10)
    retry_backoff_seconds: float | None = Field(default=None, ge=0.05, le=60.0)
    request_timeout_seconds: float | None = Field(default=None, ge=1.0, le=120.0)


class PullIntegrationStatusEnvelope(BaseModel):
    customer_id: str
    endpoint_url: str | None = None
    run_id: str | None = None
    auth_type: str = "none"
    running: bool
    status: str
    polling_interval_seconds: float | None = None
    retry_max_attempts: int | None = None
    retry_backoff_seconds: float | None = None
    request_timeout_seconds: float | None = None
    started_at: str | None = None
    updated_at: str | None = None
    last_poll_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    last_http_status: int | None = None
    total_polls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    total_ingested: int = 0
    message: str | None = None


class AlertsEnvelope(BaseModel):
    count: int
    alerts: list[dict[str, Any]]
    current_status: dict[str, Any] | None = None
    active_alert: dict[str, Any] | None = None


class CanonicalOutputResponse(BaseModel):
    schema_version: str
    timestamp: str
    cycle: int
    attribution: dict[str, Any]
    regime_memory: dict[str, Any]
    risk_assessment: dict[str, Any]
    causal_analysis: dict[str, Any]
    operational_recommendation: dict[str, Any]
    confidence: float
    explanation_text: str
    events: list[str]
    session: dict[str, Any] | None = None
    aliases: dict[str, Any] | None = None
    history_id: int | None = None
    persisted_at: str | None = None
    customer_id: str | None = None
    run_id: str | None = None


class CurrentStateEnvelope(BaseModel):
    state: CanonicalOutputResponse | None = None


class HistoryEnvelope(BaseModel):
    count: int
    history: list[CanonicalOutputResponse]


class RecommendationEnvelope(BaseModel):
    operational_recommendation: dict[str, Any] | None = None


class DecisionEnvelope(BaseModel):
    """Deprecated compatibility envelope. Prefer RecommendationEnvelope."""

    decision: dict[str, Any] | None = None


class ExplanationEnvelope(BaseModel):
    explanation_text: str | None = None


class EventsEnvelope(BaseModel):
    events: list[str] = Field(default_factory=list)
    cycle: int | None = None
    timestamp: str | None = None


class AssistantRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    mode: Literal["summary", "why_recommended", "what_changed", "pattern_similarity", "handoff"] | None = None
    history_limit: int = Field(default=20, ge=2, le=100)


class AssistantResponse(BaseModel):
    mode: str
    text: str
    grounding: dict[str, Any]
    context: dict[str, Any]



class AlertAcknowledgeRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    acknowledged_by: str | None = None


class AlertResolveRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    resolved_by: str | None = None

class ReportRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    mode: Literal["client_report", "technician_summary", "inspection_brief", "handoff_note"]
    history_limit: int = Field(default=20, ge=2, le=100)


class ReportResponse(BaseModel):
    mode: str
    report_text: str
    sections: dict[str, str]



def _alert_thresholds() -> tuple[float, float]:
    """Backward-compatible shim; implementation moved to services.alerts."""
    return service_alert_thresholds()


def _ensure_default_run(
    service: StructuralMonitoringService,
    *,
    customer_id: str | None,
) -> dict[str, Any]:
    resolved_customer = resolve_customer_id(customer_id)
    existing = service.get_active_run(customer_id=resolved_customer)
    if existing is not None:
        return existing
    return service.create_run(
        name="Default Run",
        config={"source": "api-default"},
        activate=True,
        customer_id=resolved_customer,
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


def _resolve_db_path(configured_db_path: str) -> tuple[str, bool]:
    """Return a writable SQLite path and whether persistence is available."""
    configured = str(configured_db_path or "").strip() or "neraium.db"
    if _persistence_available(configured):
        return configured, True

    fallback = "/tmp/neraium.db"
    if _persistence_available(fallback):
        logger.warning(
            "Configured NERAIUM_DB_PATH=%s is not writable; falling back to %s.",
            configured,
            fallback,
        )
        return fallback, True

    logger.error(
        "Configured NERAIUM_DB_PATH=%s and fallback=%s are not writable; using in-memory SQLite store.",
        configured,
        fallback,
    )
    return ":memory:", False


def is_api_key_valid(configured_key: str | None, provided_key: str | None) -> bool:
    if not configured_key:
        return True
    return configured_key == provided_key


def _results_envelope(results: list[dict[str, Any]], latest: dict[str, Any] | None) -> dict[str, Any]:
    return {"latest": latest, "count": len(results), "results": results}


def _compact_result_view(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return result
    trimmed = dict(result)
    # Geometry/sensor matrices are fetched from dedicated endpoints when needed.
    for key in ("sensor_values", "sensor_relationships", "geometry"):
        trimmed.pop(key, None)
    return trimmed


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(f):
        return float(default)
    return float(f)


def _parse_finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise HTTPException(status_code=400, detail=f"{field_name} must be a finite number.")
    return float(parsed)


def _parse_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be an integer.") from exc


def create_app(
    service: StructuralMonitoringService | None = None,
    *,
    max_request_body_bytes: int | None = None,
) -> FastAPI:
    api_key = os.getenv("NERAIUM_API_KEY")
    configured_db_path = os.getenv("NERAIUM_DB_PATH", "neraium.db")
    db_path, persistence_available = _resolve_db_path(configured_db_path)
    request_body_limit = (
        int(max_request_body_bytes)
        if max_request_body_bytes is not None
        else _request_body_limit_bytes()
    )

    app = FastAPI(title="Neraium SII API", version="0.1.0")
    app.add_middleware(MaxRequestBodySizeMiddleware, max_body_size=request_body_limit)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    cors_allow_origins = _cors_allow_origins()
    cors_allow_origin_regex = _cors_allow_origin_regex()
    if cors_allow_origins or cors_allow_origin_regex:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_allow_origins,
            allow_origin_regex=cors_allow_origin_regex,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=_cors_allow_headers(),
        )
        log_structured(
            logger,
            event="cors_configured",
            fields={
                "allow_origins_count": len(cors_allow_origins),
                "allow_origin_regex": bool(cors_allow_origin_regex),
            },
        )
    else:
        log_structured(
            logger,
            event="cors_disabled_same_origin_mode",
            fields={"reason": "no_allow_origins_or_origin_regex_configured"},
        )
    service_instance = service or StructuralMonitoringService(store=ResultStore(db_path=db_path))
    integration_config_path = os.getenv("NERAIUM_INTEGRATION_CONFIG_PATH")
    integration_config = load_integration_config(integration_config_path)
    app.state.integration_config_override = integration_config
    app.state.integration_config_path_override = integration_config_path
    ingest_jobs: dict[str, dict[str, Any]] = {}
    ingest_jobs_lock = threading.Lock()
    demo_jobs: dict[str, dict[str, Any]] = {}
    demo_jobs_lock = threading.Lock()
    pull_integrations: dict[str, dict[str, Any]] = {}
    pull_integrations_lock = threading.Lock()
    alerts: dict[str, list[dict[str, Any]]] = {}
    alerts_lock = threading.Lock()
    runtime_state_diagnostics: dict[str, Any] = {
        "persisted_state_enabled": False,
        "persisted_state_store": "none",
        "memory_only_state": [
            "demo_jobs",
            "cmapss_fd004_cache",
            "pull_integration_worker_threads",
            "service_engine_runtime_memory",
        ],
    }
    store_instance = getattr(service_instance, "store", None)
    persisted_state_enabled = all(
        hasattr(store_instance, method)
        for method in ("upsert_operational_state", "list_operational_state", "delete_operational_state")
    )
    runtime_state_diagnostics["persisted_state_enabled"] = bool(persisted_state_enabled)
    runtime_state_diagnostics["persisted_state_store"] = "sqlite_operational_state" if persisted_state_enabled else "none"
    alert_instability_threshold, alert_rapid_drift_delta = _alert_thresholds()
    alert_webhook_url = str(os.getenv("NERAIUM_ALERT_WEBHOOK_URL") or "").strip() or None
    alert_email_to = str(os.getenv("NERAIUM_ALERT_EMAIL_TO") or "").strip() or None
    if not persisted_state_enabled:
        log_structured(
            logger,
            event="operational_state_persistence_unavailable",
            fields={"reason": "store_missing_operational_state_methods"},
            level=logging.WARNING,
        )

    def _persist_operational_state(key: str, payload: dict[str, Any], *, customer_id: str | None = None, run_id: str | None = None) -> None:
        if not persisted_state_enabled or store_instance is None:
            return
        try:
            store_instance.upsert_operational_state(
                state_key=key,
                state=payload,
                customer_id=customer_id,
                run_id=run_id,
            )
        except Exception:
            logger.exception("Failed to persist operational state key=%s", key)

    def _delete_operational_state(key: str) -> None:
        if not persisted_state_enabled or store_instance is None:
            return
        try:
            store_instance.delete_operational_state(state_key=key)
        except Exception:
            logger.exception("Failed to delete operational state key=%s", key)

    def _record_alerts_for_customer(customer_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        resolved_customer = resolve_customer_id(customer_id)
        created: list[dict[str, Any]] = []
        with alerts_lock:
            bucket = alerts.setdefault(resolved_customer, [])
            for raw in items:
                if not isinstance(raw, dict):
                    continue
                alert = dict(raw)
                context = alert.get("context") if isinstance(alert.get("context"), dict) else {}
                trigger = alert.get("trigger") if isinstance(alert.get("trigger"), dict) else {}
                alert["customer_id"] = resolved_customer
                alert["context"] = dict(context)
                alert["trigger"] = dict(trigger)
                created.append(alert)
                bucket.append(alert)
            bucket.sort(key=lambda a: str(a.get("created_at") or ""), reverse=True)
            del bucket[200:]
            _persist_operational_state(
                f"alerts:{resolved_customer}",
                {"alerts": bucket, "customer_id": resolved_customer},
                customer_id=resolved_customer,
            )
        for alert in created:
            dispatch_alert_stubs(
                logger=logger,
                alert=alert,
                webhook_url=alert_webhook_url,
                email_to=alert_email_to,
            )
            log_structured(
                logger,
                event="alert_created",
                fields={
                    "customer_id": resolved_customer,
                    "alert_id": alert.get("id"),
                    "alert_type": alert.get("type"),
                    "severity": alert.get("severity"),
                    "run_id": (alert.get("context") or {}).get("run_id"),
                    "result_id": (alert.get("context") or {}).get("result_id"),
                },
                level=logging.WARNING,
            )
        return created

    def _process_alerts_after_ingest(
        *,
        customer_id: str,
        run_id: str | None,
        latest_result: dict[str, Any] | None,
        previous_result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        items = evaluate_alerts(
            current=latest_result,
            previous=previous_result,
            instability_threshold=alert_instability_threshold,
            rapid_drift_delta=alert_rapid_drift_delta,
            now_iso=_utc_now_iso(),
        )
        if run_id:
            for item in items:
                context = item.get("context")
                if isinstance(context, dict):
                    context.setdefault("run_id", run_id)
        return _record_alerts_for_customer(customer_id, items)

    def _normalize_content_length(request: Request) -> int | None:
        raw = request.headers.get("content-length")
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            return None
        return value if value >= 0 else None

    def _public_ingest_job(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(job.get("job_id")),
            "status": str(job.get("status", "unknown")),
            "run_id": job.get("run_id"),
            "customer_id": resolve_customer_id(job.get("customer_id")),
            "filename": str(job.get("filename") or "upload.csv"),
            "created_at": str(job.get("created_at") or _utc_now_iso()),
            "updated_at": str(job.get("updated_at") or _utc_now_iso()),
            "rows_processed": int(job.get("rows_processed", 0)),
            "rows_succeeded": int(job.get("rows_succeeded", 0)),
            "rows_failed": int(job.get("rows_failed", 0)),
            "partial_success": bool(job.get("partial_success", False)),
            "upload_bytes_received": int(job.get("upload_bytes_received", 0)),
            "upload_bytes_total": (
                int(job.get("upload_bytes_total"))
                if job.get("upload_bytes_total") is not None
                else None
            ),
            "error_samples": list(job.get("error_samples") or []),
            "message": job.get("message"),
            "latest_result": job.get("latest_result"),
        }

    def _cleanup_ingest_jobs(max_jobs: int = 300) -> None:
        removed_ids: list[str] = []
        with ingest_jobs_lock:
            if len(ingest_jobs) <= max_jobs:
                return
            completed_ids = [
                jid
                for jid, job in ingest_jobs.items()
                if str(job.get("status")) in {"completed", "partial_success", "failed"}
            ]
            completed_ids.sort(
                key=lambda jid: str(ingest_jobs[jid].get("updated_at") or ingest_jobs[jid].get("created_at") or "")
            )
            overflow = len(ingest_jobs) - max_jobs
            for jid in completed_ids[: max(0, overflow)]:
                ingest_jobs.pop(jid, None)
                removed_ids.append(jid)
        for jid in removed_ids:
            _delete_operational_state(f"ingest_job:{jid}")

    def _update_ingest_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
        with ingest_jobs_lock:
            job = ingest_jobs.get(job_id)
            if job is None:
                return None
            job.update(fields)
            job["updated_at"] = _utc_now_iso()
            if "partial_success" not in fields:
                job["partial_success"] = (
                    int(job.get("rows_succeeded", 0)) > 0 and int(job.get("rows_failed", 0)) > 0
                )
            _persist_operational_state(
                f"ingest_job:{job_id}",
                dict(job),
                customer_id=resolve_customer_id(job.get("customer_id")),
                run_id=str(job.get("run_id")) if job.get("run_id") is not None else None,
            )
            return dict(job)

    def _public_demo_job(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(job.get("job_id")),
            "status": str(job.get("status", "unknown")),
            "run_id": str(job.get("run_id") or ""),
            "customer_id": resolve_customer_id(job.get("customer_id")),
            "progress": max(0, min(100, int(job.get("progress", 0)))),
            "processed": max(0, int(job.get("processed", 0))),
            "total_frames": max(0, int(job.get("total_frames", 0))),
            "message": str(job.get("message") or ""),
            "error": job.get("error"),
            "created_at": str(job.get("created_at") or _utc_now_iso()),
            "updated_at": str(job.get("updated_at") or _utc_now_iso()),
        }

    def _update_demo_job(job_id: str, **fields: Any) -> dict[str, Any] | None:
        with demo_jobs_lock:
            job = demo_jobs.get(job_id)
            if job is None:
                return None
            job.update(fields)
            job["updated_at"] = _utc_now_iso()
            return dict(job)

    def _run_demo_seed_job(
        *,
        job_id: str,
        resolved_run: str,
        resolved_customer: str,
        payload: DemoSeedRequest,
    ) -> None:
        minutes = int(payload.minutes)
        total = max(10, min(240, minutes))
        now = datetime.now(timezone.utc)
        processed = 0
        failure_frame = None
        _update_demo_job(
            job_id,
            status="running",
            message="Seeding telemetry on server...",
            total_frames=total,
            progress=0,
            processed=0,
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        log_structured(
            logger,
            event="demo_seed_start",
            fields={
                "job_id": job_id,
                "run_id": resolved_run,
                "customer_id": resolved_customer,
                "total_frames": total,
                "profile": payload.profile,
            },
        )
        try:
            for i in range(total):
                failure_frame = i + 1
                timestamp = (now.replace(microsecond=0).timestamp() - (total - i) * 60.0)
                t = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
                if payload.profile == "watch":
                    drift_lift = 0.35 + (i / max(1, total - 1)) * 0.35
                    vib_spike = 0.55 + (i / max(1, total - 1)) * 0.45
                elif payload.profile == "critical":
                    drift_lift = 0.25 + (i / max(1, total - 1)) * 0.85
                    vib_spike = 0.8 + (i / max(1, total - 1)) * 1.8
                elif payload.profile == "sample":
                    drift_lift = 0.2 if i < total // 3 else 0.6 if i < (2 * total) // 3 else 1.0
                    vib_spike = drift_lift * 1.1
                else:
                    drift_lift = 0.12
                    vib_spike = 0.2
                p = i / max(1, total - 1)
                frame = {
                    "timestamp": t,
                    "site_id": payload.site_id,
                    "asset_id": payload.asset_id,
                    "sensor_values": _build_demo_sensor_values_row(i, p, drift_lift, vib_spike),
                    "customer_id": resolved_customer,
                }
                service_instance.ingest_frame(
                    frame,
                    run_id=resolved_run,
                    customer_id=resolved_customer,
                )
                processed += 1
                if processed % 10 == 0 or processed == total:
                    progress = int((processed / max(1, total)) * 100)
                    _update_demo_job(
                        job_id,
                        progress=progress,
                        processed=processed,
                        message=f"Seeding telemetry on server... ({processed}/{total})",
                    )
                    log_structured(
                        logger,
                        event="demo_seed_progress",
                        fields={
                            "job_id": job_id,
                            "run_id": resolved_run,
                            "customer_id": resolved_customer,
                            "processed": processed,
                            "total_frames": total,
                            "progress": progress,
                        },
                    )
        except Exception as exc:
            detail = summarize_exception_for_logs(exc)
            _update_demo_job(
                job_id,
                status="error",
                progress=int((processed / max(1, total)) * 100),
                processed=processed,
                message="Demo seed failed.",
                error=detail,
            )
            log_structured(
                logger,
                event="demo_seed_failure",
                fields={
                    "job_id": job_id,
                    "run_id": resolved_run,
                    "customer_id": resolved_customer,
                    "processed": processed,
                    "total_frames": total,
                    "failure_frame": failure_frame,
                    "error": detail,
                },
                level=logging.ERROR,
            )
            return
        _update_demo_job(
            job_id,
            status="complete",
            progress=100,
            processed=processed,
            message="Demo seeded successfully",
            error=None,
        )
        log_structured(
            logger,
            event="demo_seed_complete",
            fields={
                "job_id": job_id,
                "run_id": resolved_run,
                "customer_id": resolved_customer,
                "processed": processed,
                "total_frames": total,
            },
        )

    cmapss_fd004_cache: dict[int, list[dict[str, Any]]] = {}
    cmapss_fd004_cache_lock = threading.Lock()

    def _load_cmapss_fd004_subset(max_frames: int) -> list[dict[str, Any]]:
        limited = max(30, min(500, int(max_frames)))
        with cmapss_fd004_cache_lock:
            cached = cmapss_fd004_cache.get(limited)
            if cached is not None:
                return list(cached)

        dataset_path = Path(__file__).resolve().parents[2] / "train_FD004.txt"
        if not dataset_path.is_file():
            raise FileNotFoundError(f"CMAPSS FD004 dataset file missing at {dataset_path}")

        rows: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).replace(microsecond=0)
        sensor_keys = [f"sensor_{i}" for i in range(1, 22)]
        with dataset_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 26:
                    continue
                unit = int(float(parts[0]))
                cycle = int(float(parts[1]))
                op1 = float(parts[2])
                op2 = float(parts[3])
                op3 = float(parts[4])
                sensors = [float(v) for v in parts[5:26]]
                sensor_values = {
                    "cycle": float(cycle),
                    "op_setting_1": op1,
                    "op_setting_2": op2,
                    "op_setting_3": op3,
                }
                for idx, key in enumerate(sensor_keys):
                    sensor_values[key] = sensors[idx]
                timestamp = (now.timestamp() - max(0, limited - len(rows)) * 60.0)
                rows.append(
                    {
                        "timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                        "site_id": "nasa-cmapss-fd004",
                        "asset_id": f"engine-{unit:03d}",
                        "sensor_values": sensor_values,
                    }
                )
                if len(rows) >= limited:
                    break
        if not rows:
            raise ValueError("CMAPSS FD004 subset is empty.")
        with cmapss_fd004_cache_lock:
            cmapss_fd004_cache[limited] = list(rows)
        return rows

    def _default_pull_state(customer_id: str) -> dict[str, Any]:
        now = _utc_now_iso()
        return {
            "customer_id": customer_id,
            "endpoint_url": None,
            "run_id": None,
            "auth_type": "none",
            "running": False,
            "status": "stopped",
            "polling_interval_seconds": None,
            "retry_max_attempts": None,
            "retry_backoff_seconds": None,
            "request_timeout_seconds": None,
            "started_at": None,
            "updated_at": now,
            "last_poll_at": None,
            "last_success_at": None,
            "last_error": None,
            "last_http_status": None,
            "total_polls": 0,
            "total_failures": 0,
            "consecutive_failures": 0,
            "total_ingested": 0,
            "message": "Pull integration is stopped.",
            "_stop_event": None,
            "_thread": None,
        }

    def _public_pull_state(state: dict[str, Any] | None, *, customer_id: str) -> dict[str, Any]:
        base = _default_pull_state(customer_id)
        private_keys = {"username", "password", "token"}
        if state is None:
            return {
                k: v
                for k, v in base.items()
                if not k.startswith("_") and k not in private_keys
            }
        merged = dict(base)
        merged.update(state)
        return {
            k: v
            for k, v in merged.items()
            if not k.startswith("_") and k not in private_keys
        }

    def _persist_pull_state(customer_id: str, state: dict[str, Any]) -> None:
        public_state = _public_pull_state(state, customer_id=customer_id)
        public_state["resume_on_startup"] = bool(public_state.get("running"))
        _persist_operational_state(
            f"pull_integration:{customer_id}",
            public_state,
            customer_id=customer_id,
            run_id=str(public_state.get("run_id")) if public_state.get("run_id") is not None else None,
        )

    def _validate_endpoint_url(endpoint_url: str) -> str:
        text = str(endpoint_url or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="endpoint_url is required.")
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(
                status_code=400,
                detail="endpoint_url must be a valid http(s) URL.",
            )
        return text

    def _pull_auth_header(state: dict[str, Any]) -> str | None:
        auth_type = str(state.get("auth_type") or "none")
        if auth_type == "bearer":
            token = str(state.get("token") or "").strip()
            if not token:
                raise ValueError("Bearer token is empty.")
            return f"Bearer {token}"
        if auth_type == "basic":
            username = str(state.get("username") or "")
            password = str(state.get("password") or "")
            raw = f"{username}:{password}".encode("utf-8")
            return "Basic " + base64.b64encode(raw).decode("ascii")
        return None

    def _coerce_pull_items(payload: Any, *, customer_id: str) -> list[dict[str, Any]]:
        cfg_override = getattr(app.state, "integration_config_override", None)
        if isinstance(cfg_override, dict):
            cfg = cfg_override
        else:
            path_override = getattr(app.state, "integration_config_path_override", None)
            path = str(path_override or "").strip() or os.getenv("NERAIUM_INTEGRATION_CONFIG_PATH")
            cfg = load_integration_config(path)
        try:
            rows = apply_integration_mapping(
                payload,
                customer_id=customer_id,
                config=cfg,
            )
        except IntegrationMappingError as exc:
            raise ValueError(str(exc)) from exc
        return rows

    def _fetch_pull_payload(state: dict[str, Any]) -> tuple[int, Any]:
        endpoint_url = str(state.get("endpoint_url") or "").strip()
        timeout_s = float(state.get("request_timeout_seconds") or 10.0)
        headers = {"Accept": "application/json"}
        auth_header = _pull_auth_header(state)
        if auth_header:
            headers["Authorization"] = auth_header
        req = urllib.request.Request(endpoint_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            status_code = int(response.getcode() or 0)
            body_bytes = response.read()
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(f"Upstream returned HTTP {status_code}.")
        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Upstream response is not valid JSON.") from exc
        return status_code, payload

    def _ingest_pull_items(*, rows: list[dict[str, Any]], run_id: str, customer_id: str) -> int:
        if not rows:
            return 0
        if len(rows) == 1:
            service_instance.ingest_payload(rows[0], run_id=run_id, customer_id=customer_id)
            return 1
        results = service_instance.ingest_batch(rows, run_id=run_id, customer_id=customer_id)
        return len(results)

    def _stop_pull_integration(customer_id: str, *, reason: str) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        stop_event: threading.Event | None = None
        thread: threading.Thread | None = None
        with pull_integrations_lock:
            state = pull_integrations.get(resolved_customer)
            if state is None:
                return _public_pull_state(None, customer_id=resolved_customer)
            stop_event = state.get("_stop_event")
            thread = state.get("_thread")
            state["running"] = False
            state["status"] = "stopped"
            state["message"] = reason
            state["updated_at"] = _utc_now_iso()
            _persist_pull_state(resolved_customer, state)
        if isinstance(stop_event, threading.Event):
            stop_event.set()
        if isinstance(thread, threading.Thread) and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with pull_integrations_lock:
            final_state = pull_integrations.get(resolved_customer)
            return _public_pull_state(final_state, customer_id=resolved_customer)

    def _start_pull_worker(customer_id: str) -> None:
        resolved_customer = resolve_customer_id(customer_id)

        def _worker() -> None:
            while True:
                with pull_integrations_lock:
                    state = pull_integrations.get(resolved_customer)
                    if state is None:
                        return
                    stop_event = state.get("_stop_event")
                    is_running = bool(state.get("running"))
                    endpoint_url = str(state.get("endpoint_url") or "")
                    poll_interval = _safe_float(state.get("polling_interval_seconds"), 30.0)
                    retry_attempts = max(1, _parse_int(state.get("retry_max_attempts") or 3, field_name="retry_max_attempts"))
                    retry_backoff = _safe_float(state.get("retry_backoff_seconds"), 1.0)
                    run_id = str(state.get("run_id") or "")
                    # Do not clobber "stopped" / running=False set by stop endpoint before we exit.
                    if is_running:
                        state["status"] = "running"
                        state["updated_at"] = _utc_now_iso()
                        state["message"] = "Polling upstream API."
                        _persist_pull_state(resolved_customer, state)
                if not is_running or not isinstance(stop_event, threading.Event):
                    return
                if stop_event.is_set():
                    return
                if not endpoint_url or not run_id:
                    with pull_integrations_lock:
                        current = pull_integrations.get(resolved_customer)
                        if current is not None:
                            current["running"] = False
                            current["status"] = "error"
                            current["message"] = "Integration misconfigured: missing endpoint or run_id."
                            current["last_error"] = current["message"]
                            current["updated_at"] = _utc_now_iso()
                            _persist_pull_state(resolved_customer, current)
                    return

                success = False
                last_error = ""
                for attempt in range(1, max(1, retry_attempts) + 1):
                    with pull_integrations_lock:
                        current = pull_integrations.get(resolved_customer)
                        if current is None:
                            return
                        current["last_poll_at"] = _utc_now_iso()
                        current["total_polls"] = int(current.get("total_polls", 0)) + 1
                        current["updated_at"] = _utc_now_iso()
                        _persist_pull_state(resolved_customer, current)
                    try:
                        http_status, payload = _fetch_pull_payload(state)
                        rows = _coerce_pull_items(payload, customer_id=resolved_customer)
                        ingested = _ingest_pull_items(
                            rows=rows,
                            run_id=run_id,
                            customer_id=resolved_customer,
                        )
                        now = _utc_now_iso()
                        with pull_integrations_lock:
                            current = pull_integrations.get(resolved_customer)
                            if current is None:
                                return
                            current["last_http_status"] = int(http_status)
                            current["last_error"] = None
                            current["last_success_at"] = now
                            current["consecutive_failures"] = 0
                            current["total_ingested"] = int(current.get("total_ingested", 0)) + int(ingested)
                            current["status"] = "running"
                            current["message"] = f"Last poll ingested {ingested} item(s)."
                            current["updated_at"] = now
                            _persist_pull_state(resolved_customer, current)
                        log_structured(
                            logger,
                            event="pull_integration_poll_success",
                            fields={
                                "customer_id": resolved_customer,
                                "run_id": run_id,
                                "ingested_items": int(ingested),
                                "http_status": int(http_status),
                            },
                            level=logging.INFO,
                        )
                        success = True
                        break
                    except Exception as exc:
                        last_error = str(exc)
                        with pull_integrations_lock:
                            current = pull_integrations.get(resolved_customer)
                            if current is None:
                                return
                            current["total_failures"] = int(current.get("total_failures", 0)) + 1
                            current["consecutive_failures"] = int(current.get("consecutive_failures", 0)) + 1
                            current["status"] = "error"
                            current["last_error"] = last_error
                            current["message"] = (
                                f"Poll attempt {attempt}/{retry_attempts} failed: {last_error}"
                            )
                            current["updated_at"] = _utc_now_iso()
                            _persist_pull_state(resolved_customer, current)
                        log_structured(
                            logger,
                            event="pull_integration_poll_failure",
                            fields={
                                "customer_id": resolved_customer,
                                "run_id": run_id,
                                "attempt": attempt,
                                "retry_attempts": retry_attempts,
                                "error": last_error,
                                **summarize_exception_for_logs(exc),
                            },
                            level=logging.WARNING,
                        )
                        if attempt >= retry_attempts:
                            break
                        delay = max(0.05, _safe_float(retry_backoff, 1.0)) * (2 ** (attempt - 1))
                        if stop_event.wait(delay):
                            return

                if stop_event.wait(max(0.2, _safe_float(poll_interval, 30.0))):
                    return
                if not success:
                    with pull_integrations_lock:
                        current = pull_integrations.get(resolved_customer)
                        if current is not None:
                            current["message"] = "Polling will continue after previous failure."
                            current["updated_at"] = _utc_now_iso()
                            _persist_pull_state(resolved_customer, current)

        worker = threading.Thread(target=_worker, daemon=True, name=f"pull-integration-{resolved_customer}")
        with pull_integrations_lock:
            state = pull_integrations.get(resolved_customer)
            if state is None:
                return
            state["_thread"] = worker
        worker.start()

    def _restore_persisted_operational_state() -> None:
        if not persisted_state_enabled or store_instance is None:
            return
        restored_alert_customers = 0
        restored_ingest_jobs = 0
        restored_pull_integrations = 0
        try:
            for row in store_instance.list_operational_state(key_prefix="alerts:"):
                state = row.get("state")
                if not isinstance(state, dict):
                    continue
                customer = resolve_customer_id(state.get("customer_id") or row.get("customer_id"))
                items = state.get("alerts")
                if not isinstance(items, list):
                    continue
                with alerts_lock:
                    alerts[customer] = [dict(x) for x in items if isinstance(x, dict)][:200]
                restored_alert_customers += 1
        except Exception:
            logger.exception("Failed restoring persisted alerts state")
        try:
            for row in store_instance.list_operational_state(key_prefix="ingest_job:"):
                state = row.get("state")
                if not isinstance(state, dict):
                    continue
                job_id = str(state.get("job_id") or "")
                if not job_id:
                    continue
                status = str(state.get("status") or "unknown")
                # Upload/processing cannot resume safely across restart.
                if status in {"uploading", "queued", "processing"}:
                    state["status"] = "failed"
                    state["message"] = "Job interrupted by process restart before completion."
                    state["updated_at"] = _utc_now_iso()
                    _persist_operational_state(
                        f"ingest_job:{job_id}",
                        state,
                        customer_id=resolve_customer_id(state.get("customer_id")),
                        run_id=str(state.get("run_id")) if state.get("run_id") is not None else None,
                    )
                with ingest_jobs_lock:
                    ingest_jobs[job_id] = dict(state)
                restored_ingest_jobs += 1
        except Exception:
            logger.exception("Failed restoring persisted ingest job state")
        try:
            for row in store_instance.list_operational_state(key_prefix="pull_integration:"):
                state = row.get("state")
                if not isinstance(state, dict):
                    continue
                customer = resolve_customer_id(state.get("customer_id") or row.get("customer_id"))
                merged = _default_pull_state(customer)
                merged.update({k: v for k, v in state.items() if not str(k).startswith("_")})
                should_resume = bool(state.get("resume_on_startup"))
                merged["running"] = should_resume
                merged["status"] = "running" if should_resume else str(merged.get("status") or "stopped")
                merged["message"] = (
                    "Pull integration resumed after restart." if should_resume else str(merged.get("message") or "Pull integration is stopped.")
                )
                merged["_stop_event"] = threading.Event() if should_resume else None
                merged["_thread"] = None
                merged["updated_at"] = _utc_now_iso()
                with pull_integrations_lock:
                    pull_integrations[customer] = merged
                _persist_pull_state(customer, merged)
                if should_resume:
                    _start_pull_worker(customer)
                restored_pull_integrations += 1
        except Exception:
            logger.exception("Failed restoring persisted pull integration state")
        runtime_state_diagnostics["restored_state_counts"] = {
            "alert_customers": restored_alert_customers,
            "ingest_jobs": restored_ingest_jobs,
            "pull_integrations": restored_pull_integrations,
        }
        log_structured(
            logger,
            event="operational_state_restore_complete",
            fields=runtime_state_diagnostics.get("restored_state_counts") or {},
            level=logging.INFO,
        )

    async def _stream_upload_to_tempfile(upload: UploadFile, target_path: Path, job_id: str) -> int:
        bytes_received = 0
        chunk_size = max(16 * 1024, DEFAULT_UPLOAD_STREAM_CHUNK_BYTES)
        try:
            with target_path.open("wb") as out:
                while True:
                    chunk = await upload.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_received += len(chunk)
                    _update_ingest_job(
                        job_id,
                        status="uploading",
                        upload_bytes_received=bytes_received,
                        message=f"Uploading CSV ({bytes_received} bytes received)...",
                    )
                out.flush()
        finally:
            await upload.close()
        return bytes_received

    def _start_ingest_job_worker(
        *,
        job_id: str,
        temp_path: str,
        run_id: str,
        customer_id: str,
        column_mapping: dict[str, Any] | None = None,
    ) -> None:
        def _worker() -> None:
            _update_ingest_job(
                job_id,
                status="processing",
                message="Upload complete. Ingest processing started.",
            )
            try:
                def _on_progress(progress: dict[str, Any]) -> None:
                    progress_status = str(progress.get("status") or "processing")
                    _update_ingest_job(
                        job_id,
                        status=progress_status if progress_status in {"processing", "completed"} else "processing",
                        rows_processed=int(progress.get("rows_processed", 0)),
                        rows_succeeded=int(progress.get("rows_succeeded", 0)),
                        rows_failed=int(progress.get("rows_failed", 0)),
                        error_samples=list(progress.get("error_samples") or []),
                        message=progress.get("message"),
                    )

                with Path(temp_path).open("r", encoding="utf-8", newline="") as stream:
                    summary = service_instance.ingest_csv_stream(
                        stream,
                        run_id=run_id,
                        customer_id=customer_id,
                        column_mapping=column_mapping,
                        max_error_samples=DEFAULT_INGEST_JOB_MAX_ERROR_SAMPLES,
                        progress_callback=_on_progress,
                    )
                final_status = "completed"
                if int(summary.get("rows_failed", 0)) > 0:
                    final_status = "partial_success" if int(summary.get("rows_succeeded", 0)) > 0 else "failed"
                _update_ingest_job(
                    job_id,
                    status=final_status,
                    rows_processed=int(summary.get("rows_processed", 0)),
                    rows_succeeded=int(summary.get("rows_succeeded", 0)),
                    rows_failed=int(summary.get("rows_failed", 0)),
                    partial_success=bool(summary.get("partial_success", False)),
                    error_samples=list(summary.get("error_samples") or []),
                    latest_result=summary.get("latest_result"),
                    message=summary.get("message"),
                )
                latest_from_summary = summary.get("latest_result")
                previous_for_alerts: dict[str, Any] | None = None
                if isinstance(latest_from_summary, dict):
                    recent_for_alerts = service_instance.list_recent_results(
                        limit=2,
                        run_id=run_id,
                        customer_id=customer_id,
                    )
                    if len(recent_for_alerts) >= 2:
                        previous_for_alerts = recent_for_alerts[1]
                    _process_alerts_after_ingest(
                        customer_id=customer_id,
                        run_id=run_id,
                        latest_result=latest_from_summary,
                        previous_result=previous_for_alerts,
                    )
                log_structured(
                    logger,
                    event="ingest_job_completed",
                    fields={
                        "job_id": job_id,
                        "run_id": run_id,
                        "customer_id": customer_id,
                        "rows_processed": int(summary.get("rows_processed", 0)),
                        "rows_succeeded": int(summary.get("rows_succeeded", 0)),
                        "rows_failed": int(summary.get("rows_failed", 0)),
                        "partial_success": bool(summary.get("partial_success", False)),
                    },
                    level=logging.INFO,
                )
            except Exception as exc:
                message = actionable_validation_detail(str(exc))
                _update_ingest_job(job_id, status="failed", message=message)
                log_structured(
                    logger,
                    event="ingest_job_failed",
                    fields={
                        "job_id": job_id,
                        "run_id": run_id,
                        "customer_id": customer_id,
                        **summarize_exception_for_logs(exc),
                    },
                    level=logging.ERROR,
                )
            finally:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    logger.debug("Unable to remove temp upload file: %s", temp_path, exc_info=True)
                _cleanup_ingest_jobs()

        threading.Thread(target=_worker, daemon=True, name=f"ingest-job-{job_id}").start()

    _restore_persisted_operational_state()

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if not is_api_key_valid(api_key, x_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    app.include_router(
        build_health_router(
            service_instance=service_instance,
            api_key=api_key,
            persistence_available=persistence_available,
            app_version=app.version,
            resolve_customer_id=resolve_customer_id,
            runtime_state_diagnostics_provider=lambda: dict(runtime_state_diagnostics),
            health_response_model=HealthResponse,
            client_error_model=ClientErrorReport,
        )
    )
    app.include_router(
        build_geometry_router(
            service_instance=service_instance,
            resolve_customer_id=resolve_customer_id,
            geometry_envelope_model=GeometryEnvelope,
        )
    )
    app.include_router(
        build_alerts_router(
            service_instance=service_instance,
            require_api_key=require_api_key,
            resolve_customer_id=resolve_customer_id,
            request_run_id_or_active=request_run_id_or_active,
            utc_now_iso=_utc_now_iso,
            alerts=alerts,
            alerts_lock=alerts_lock,
            record_alerts_for_customer=_record_alerts_for_customer,
            alerts_envelope_model=AlertsEnvelope,
            action_response_model=ActionResponse,
            alert_ack_model=AlertAcknowledgeRequest,
            alert_resolve_model=AlertResolveRequest,
        )
    )

    @app.post("/runs", response_model=RunEnvelope)
    def create_run(
        payload: CreateRunRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            run = service_instance.create_run(
                name=payload.name.strip(),
                config=dict(payload.config),
                activate=bool(payload.activate),
                customer_id=resolve_customer_id(customer_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"run": run}

    @app.post("/runs/activate", response_model=RunEnvelope)
    def activate_run(
        payload: ActivateRunRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            run = service_instance.activate_run(
                payload.run_id.strip(),
                customer_id=resolve_customer_id(customer_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run": run}

    @app.get("/runs", response_model=RunsEnvelope)
    def list_runs(
        limit: int = Query(50, ge=1, le=500),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        runs = service_instance.list_runs(limit=limit, customer_id=resolved_customer)
        active = service_instance.get_active_run(customer_id=resolved_customer)
        return {"active_run": active, "count": len(runs), "runs": runs}

    @app.get("/runs/active", response_model=RunEnvelope)
    def get_active_run(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = service_instance.get_active_run(customer_id=resolve_customer_id(customer_id))
        if run is None:
            return {"run": None}
        return {"run": run}

    @app.get("/runs/{run_id}", response_model=RunEnvelope)
    def get_run(run_id: str, customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = service_instance.get_run(run_id, customer_id=resolve_customer_id(customer_id))
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return {"run": run}

    @app.patch("/runs/{run_id}", response_model=RunEnvelope)
    def update_run(
        run_id: str,
        payload: UpdateRunRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            run = service_instance.update_run(
                run_id,
                name=payload.name,
                config=payload.config,
                status=payload.status,
                customer_id=resolve_customer_id(customer_id),
            )
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if "Unknown run_id" in detail else 400
            raise HTTPException(status_code=status_code, detail=detail)
        return {"run": run}

    @app.post("/runs/{run_id}/activate", response_model=RunEnvelope)
    def activate_run_path(
        run_id: str,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            run = service_instance.activate_run(
                run_id.strip(),
                customer_id=resolve_customer_id(customer_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run": run}

    @app.get("/runs/{run_id}/baseline", response_model=dict)
    def get_run_baseline(
        run_id: str,
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        run = service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return service_instance.get_baseline_info_for_run(run_id, customer_id=resolved_customer)

    @app.post("/runs/{run_id}/baseline/reset", response_model=ActionResponse)
    def reset_run_baseline(
        run_id: str,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, bool]:
        resolved_customer = resolve_customer_id(customer_id)
        run = service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        service_instance.reset_baseline_for_run(run_id, customer_id=resolved_customer)
        return {"ok": True}

    @app.post("/runs/{run_id}/baseline/lock", response_model=RunEnvelope)
    def lock_run_baseline(
        run_id: str,
        payload: LockBaselineRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        try:
            service_instance.lock_baseline_for_run(
                run_id, locked=payload.locked, customer_id=resolved_customer,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        run = service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return {"run": run}

    @app.post("/ingest", response_model=ResultsEnvelope)
    def ingest(
        payload: IngestRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        logger.info("ingest endpoint called")
        try:
            resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
            resolved = resolve_run_id_with_default(
                service_instance,
                run_id,
                customer_id=resolved_customer,
                )
            result = service_instance.ingest_payload(
                payload.model_dump(exclude_none=True),
                run_id=resolved,
                customer_id=resolved_customer,
            )
            previous_result = None
            recent_for_alerts = service_instance.list_recent_results(
                limit=2,
                run_id=resolved,
                customer_id=resolved_customer,
            )
            if len(recent_for_alerts) >= 2:
                previous_result = recent_for_alerts[1]
            _process_alerts_after_ingest(
                customer_id=resolved_customer,
                run_id=resolved,
                latest_result=result,
                previous_result=previous_result,
            )
        except ValueError as e:
            logger.warning("validation failure ingest: %s", e)
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(e)))
        return _results_envelope([result], latest=result)

    @app.post("/ingest/frame", response_model=CanonicalOutputResponse)
    def ingest_frame(
        payload: IngestFrameRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
            resolved_run = resolve_run_id_with_default(
                service_instance,
                run_id,
                customer_id=resolved_customer,
                )
            return service_instance.ingest_frame(
                payload.model_dump(exclude_none=True),
                run_id=resolved_run,
                customer_id=resolved_customer,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(exc)))

    @app.post("/demo/seed/start")
    def demo_seed_start(
        payload: DemoSeedRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
        resolved_run = resolve_run_id_with_default(
            service_instance,
            run_id or payload.run_id,
            customer_id=resolved_customer,
            )
        job_id = str(uuid4())
        now = _utc_now_iso()
        job = {
            "job_id": job_id,
            "status": "pending",
            "run_id": resolved_run,
            "customer_id": resolved_customer,
            "progress": 0,
            "processed": 0,
            "total_frames": int(payload.minutes),
            "message": "Preparing demo run...",
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        with demo_jobs_lock:
            demo_jobs[job_id] = job
        worker = threading.Thread(
            target=_run_demo_seed_job,
            kwargs={
                "job_id": job_id,
                "resolved_run": resolved_run,
                "resolved_customer": resolved_customer,
                "payload": payload,
            },
            daemon=True,
        )
        worker.start()
        return {
            "status": "started",
            "job_id": job_id,
            "run_id": resolved_run,
            "message": "Demo seeding started.",
        }

    @app.get("/demo/seed/status")
    def demo_seed_status(
        job_id: str = Query(..., min_length=1),
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        with demo_jobs_lock:
            job = demo_jobs.get(job_id)
        if job is None:
            return {
                "status": "error",
                "job_id": job_id,
                "progress": 0,
                "run_id": "",
                "processed": 0,
                "total_frames": 0,
                "message": "Demo seed job not found.",
                "error": "job_not_found",
            }
        return _public_demo_job(job)

    @app.post("/demo/cmapss/start")
    def demo_cmapss_start(
        payload: DemoCmapssStartRequest | None = None,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        request_payload = payload or DemoCmapssStartRequest()
        resolved_customer = resolve_customer_id(customer_id or request_payload.customer_id)
        log_structured(
            logger,
            event="demo_cmapss_route_entry",
            fields={
                "customer_id": resolved_customer,
                "requested_max_frames": int(request_payload.max_frames),
            },
            level=logging.INFO,
        )
        run = service_instance.create_run(
            name=f"NASA CMAPSS FD004 Demo {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            config={
                "source": "nasa-cmapss-fd004",
                "dataset": "NASA CMAPSS FD004",
                "demo": "cmapss_fd004",
                "historical_run_replay": True,
            },
            activate=True,
            customer_id=resolved_customer,
            )
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise HTTPException(status_code=500, detail="Failed to create demo run.")
        log_structured(
            logger,
            event="demo_cmapss_run_created",
            fields={"run_id": run_id, "customer_id": resolved_customer},
            level=logging.INFO,
        )
        try:
            rows = _load_cmapss_fd004_subset(request_payload.max_frames)
            log_structured(
                logger,
                event="demo_cmapss_processing_start",
                fields={"run_id": run_id, "customer_id": resolved_customer, "rows": len(rows)},
                level=logging.INFO,
            )
            payload_rows = [
                {**row, "customer_id": resolved_customer}
                for row in rows
            ]
            results = service_instance.ingest_batch(
                payload_rows,
                run_id=run_id,
                customer_id=resolved_customer,
            )
            latest = service_instance.get_latest_result(run_id=run_id, customer_id=resolved_customer)
            log_structured(
                logger,
                event="demo_cmapss_processing_complete",
                fields={
                    "run_id": run_id,
                    "customer_id": resolved_customer,
                    "processed": len(results),
                    "latest_result_available": latest is not None,
                },
                level=logging.INFO,
            )
        except HTTPException:
            raise
        except Exception as exc:
            detail = summarize_exception_for_logs(exc)
            log_structured(
                logger,
                event="demo_cmapss_start_failure",
                fields={"run_id": run_id, "customer_id": resolved_customer, "error": detail},
                level=logging.ERROR,
            )
            raise HTTPException(status_code=500, detail=f"Failed to run NASA CMAPSS FD004 demo: {detail}")
        return {
            "status": "ok",
            "run_id": run_id,
            "processed": len(results),
            "demo": "cmapss_fd004",
        }

    @app.get("/state", response_model=CurrentStateEnvelope)
    def get_state(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        state = service_instance.get_current_state(
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        return {"state": state}

    @app.get("/history", response_model=HistoryEnvelope)
    def get_history(
        limit: int = Query(default=100, ge=1, le=1000),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        history = service_instance.get_recent_history(
            limit=limit,
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        return {"count": len(history), "history": history}

    @app.get("/recommendation", response_model=RecommendationEnvelope)
    @app.get("/recommendations/latest", response_model=RecommendationEnvelope)
    def get_recommendation(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        recommendation = service_instance.get_latest_recommendation(
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        return {"operational_recommendation": recommendation}

    @app.get("/decision", response_model=DecisionEnvelope, deprecated=True)
    def get_decision(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        decision = service_instance.get_latest_decision(
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        return {"decision": decision}

    @app.get("/explanation", response_model=ExplanationEnvelope)
    def get_explanation(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        explanation_text = service_instance.get_latest_explanation(
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        return {"explanation_text": explanation_text}

    @app.get("/events/latest", response_model=EventsEnvelope)
    def get_events_latest(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        state = service_instance.get_current_state(
            run_id=resolved_run,
            customer_id=resolved_customer,
            )
        if not isinstance(state, dict):
            return {"events": [], "cycle": None, "timestamp": None}
        events = state.get("events")
        return {
            "events": list(events) if isinstance(events, list) else [],
            "cycle": state.get("cycle"),
            "timestamp": state.get("timestamp"),
        }

    @app.post("/assistant/summary", response_model=AssistantResponse)
    def assistant_summary(
        payload: AssistantRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run = resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
        return service_instance.generate_assistant_response(
            mode="summary",
            run_id=resolved_run,
            customer_id=resolved_customer,
            history_limit=payload.history_limit,
        )

    @app.post("/assistant/handoff", response_model=AssistantResponse)
    def assistant_handoff(
        payload: AssistantRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run = resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
        return service_instance.generate_assistant_response(
            mode="handoff",
            run_id=resolved_run,
            customer_id=resolved_customer,
            history_limit=payload.history_limit,
        )

    @app.post("/assistant/explain", response_model=AssistantResponse)
    def assistant_explain(
        payload: AssistantRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run = resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
        mode = payload.mode or "why_recommended"
        if mode not in {"why_recommended", "what_changed", "pattern_similarity"}:
            raise HTTPException(
                status_code=400,
                detail="mode must be one of: why_recommended, what_changed, pattern_similarity",
            )
        return service_instance.generate_assistant_response(
            mode=mode,
            run_id=resolved_run,
            customer_id=resolved_customer,
            history_limit=payload.history_limit,
        )

    @app.post("/assistant/report", response_model=ReportResponse)
    def assistant_report(
        payload: ReportRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run = resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
        report = service_instance.generate_report_response(
            mode=payload.mode,
            run_id=resolved_run,
            customer_id=resolved_customer,
            history_limit=payload.history_limit,
        )
        return {
            "mode": report.get("mode", payload.mode),
            "report_text": str(report.get("report_text") or ""),
            "sections": {k: str(v) for k, v in (report.get("sections") or {}).items()},
        }

    @app.post("/assistant/report/export")
    def assistant_report_export(
        payload: ReportRequest,
        format: Literal["txt", "md"] = Query(default="txt"),
        _: None = Depends(require_api_key),
    ) -> PlainTextResponse:
        resolved_customer = resolve_customer_id(payload.customer_id)
        resolved_run = resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
        report = service_instance.generate_report_response(
            mode=payload.mode,
            run_id=resolved_run,
            customer_id=resolved_customer,
            history_limit=payload.history_limit,
        )
        text = str(report.get("report_text") or "")
        filename = f"{payload.mode}_{resolved_run or 'run'}.{format}"
        media_type = "text/markdown; charset=utf-8" if format == "md" else "text/plain; charset=utf-8"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return PlainTextResponse(content=text, media_type=media_type, headers=headers)

    @app.post("/ingest/batch", response_model=ResultsEnvelope)
    def ingest_batch(
        payload: BatchIngestRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        batch_size = len(payload.items)
        log_structured(
            logger,
            event="ingest_batch_request_received",
            fields={
                "batch_size": batch_size,
                "query_run_id": run_id,
                "query_customer_id": customer_id,
            },
            level=logging.INFO,
        )
        resolved: str | None = None
        results: list[dict[str, Any]] = []
        resolved_customer: str | None = None
        try:
            payload_customer = None
            if payload.items:
                payload_customer = payload.items[0].customer_id
            resolved_customer = resolve_customer_id(customer_id or payload_customer)
            resolved = resolve_run_id_with_default(
                service_instance,
                run_id,
                customer_id=resolved_customer,
                )
            log_structured(
                logger,
                event="ingest_batch_starting_ingest",
                fields={
                    "batch_size": batch_size,
                    "run_id": resolved,
                    "customer_id": resolved_customer,
                },
                level=logging.INFO,
            )
            results = service_instance.ingest_batch(
                [item.model_dump(exclude_none=True) for item in payload.items],
                run_id=resolved,
                customer_id=resolved_customer,
            )
            if results:
                previous_result = None
                recent_for_alerts = service_instance.list_recent_results(
                    limit=2,
                    run_id=resolved,
                    customer_id=resolved_customer,
                )
                if len(recent_for_alerts) >= 2:
                    previous_result = recent_for_alerts[1]
                _process_alerts_after_ingest(
                    customer_id=resolved_customer,
                    run_id=resolved,
                    latest_result=results[-1],
                    previous_result=previous_result,
                )
        except ValueError as e:
            detail = actionable_validation_detail(str(e))
            log_structured(
                logger,
                event="ingest_batch_validation_failure",
                fields={
                    "batch_size": batch_size,
                    "run_id": resolved,
                    "customer_id": resolved_customer,
                    "error_type": type(e).__name__,
                    "detail": detail,
                },
                level=logging.WARNING,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "ingest failed",
                    "detail": detail,
                    "run_id": resolved,
                    "customer_id": resolved_customer,
                    "batch_size": batch_size,
                    "processed": 0,
                    "count": 0,
                    "results": [],
                    "latest": None,
                },
            )
        except Exception as e:  # noqa: BLE001 - ensure the route never leaks unhandled exceptions.
            log_structured(
                logger,
                event="ingest_batch_unhandled_failure",
                fields={
                    "batch_size": batch_size,
                    "run_id": resolved,
                    "customer_id": resolved_customer,
                    "error_type": type(e).__name__,
                    "detail": str(e),
                },
                level=logging.ERROR,
            )
            logger.exception("ingest_batch unhandled exception")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "ingest failed",
                    "detail": str(e),
                    "run_id": resolved,
                    "customer_id": resolved_customer,
                    "batch_size": batch_size,
                    "processed": 0,
                    "count": 0,
                    "results": [],
                    "latest": None,
                },
            )
        log_structured(
            logger,
            event="ingest_batch_completed",
            fields={
                "batch_size": batch_size,
                "processed": len(results),
                "run_id": resolved,
                "customer_id": resolved_customer,
            },
            level=logging.INFO,
        )
        envelope = _results_envelope(results, latest=results[-1] if results else None)
        envelope["status"] = "ok"
        envelope["processed"] = len(results)
        envelope["run_id"] = resolved
        return envelope

    @app.post("/ingest/csv", response_model=ResultsEnvelope)
    def ingest_csv(
        payload: CsvIngestRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        logger.info("ingest_csv endpoint called")
        try:
            resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
            resolved = resolve_run_id_with_default(
                service_instance,
                run_id,
                customer_id=resolved_customer,
                )
            results = service_instance.ingest_csv(
                payload.csv_text,
                run_id=resolved,
                customer_id=resolved_customer,
                column_mapping=payload.column_mapping.model_dump() if payload.column_mapping else None,
            )
            if results:
                previous_result = None
                recent_for_alerts = service_instance.list_recent_results(
                    limit=2,
                    run_id=resolved,
                    customer_id=resolved_customer,
                )
                if len(recent_for_alerts) >= 2:
                    previous_result = recent_for_alerts[1]
                _process_alerts_after_ingest(
                    customer_id=resolved_customer,
                    run_id=resolved,
                    latest_result=results[-1],
                    previous_result=previous_result,
                )
        except ValueError as e:
            logger.warning("validation failure ingest_csv: %s", e)
            raise HTTPException(status_code=400, detail=actionable_validation_detail(str(e)))
        return _results_envelope(results, latest=results[-1] if results else None)

    @app.post("/ingest/csv/preview", response_model=CsvPreviewResponse)
    def ingest_csv_preview(
        payload: CsvPreviewRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Infer semantic column roles from an arbitrary CSV header + sample rows."""
        _ = resolve_customer_id(customer_id)
        headers, rows = parse_csv_sample_for_mapping(payload.csv_sample, max_rows=16)
        if not headers:
            raise HTTPException(
                status_code=400,
                detail="CSV sample has no header row.",
            )
        mapping, issues, _debug = infer_semantic_mapping(headers, sample_rows=rows)
        warnings = [i for i in issues if "Confirm" in i or "Multiple" in i]
        infer_blocking = [i for i in issues if i not in warnings]
        hard_issues: list[str] = list(infer_blocking)
        if mapping is not None:
            errs = validate_mapping(mapping, headers)
            if errs:
                hard_issues.extend(errs)
                mapping = None
        requires_confirmation = mapping is None or len(warnings) > 0
        return CsvPreviewResponse(
            headers=headers,
            suggested_mapping=mapping.to_dict() if mapping else None,
            issues=hard_issues,
            warnings=warnings,
            requires_confirmation=requires_confirmation,
        )

    @app.post("/ingest/csv/upload", response_model=IngestJobEnvelope)
    async def ingest_csv_upload(
        request: Request,
        file: UploadFile = File(...),
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        mapping: str | None = Form(default=None),
    ) -> dict[str, Any]:
        filename = str(file.filename or "upload.csv")
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Upload must be a .csv file.",
            )
        resolved_customer = resolve_customer_id(customer_id)
        resolved_run = resolve_run_id_with_default(
            service_instance,
            run_id,
            customer_id=resolved_customer,
            )
        column_mapping: dict[str, Any] | None = None
        if mapping:
            try:
                parsed = json.loads(mapping)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="column_mapping must be valid JSON (timestamp, asset_id, optional site_id, sensor_columns).",
                ) from exc
            try:
                column_mapping = CsvColumnMappingPayload.model_validate(parsed).model_dump()
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid column_mapping: " + str(exc),
                ) from exc
        content_length = _normalize_content_length(request)
        if content_length is not None and content_length > request_body_limit:
            max_mb = request_body_limit / (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Request body too large (max {max_mb:.1f}MB).",
            )

        fd, temp_path = tempfile.mkstemp(prefix="neraium_ingest_", suffix=".csv")
        os.close(fd)
        job_id = f"ingest_{uuid4().hex[:16]}"
        created_at = _utc_now_iso()
        initial_job = {
            "job_id": job_id,
            "status": "uploading",
            "run_id": resolved_run,
            "customer_id": resolved_customer,
            "filename": filename,
            "created_at": created_at,
            "updated_at": created_at,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "partial_success": False,
            "upload_bytes_received": 0,
            "upload_bytes_total": content_length,
            "error_samples": [],
            "message": "Upload started.",
            "latest_result": None,
        }
        with ingest_jobs_lock:
            ingest_jobs[job_id] = initial_job
        _persist_operational_state(
            f"ingest_job:{job_id}",
            initial_job,
            customer_id=resolved_customer,
            run_id=resolved_run,
        )

        try:
            bytes_received = await _stream_upload_to_tempfile(file, Path(temp_path), job_id)
        except Exception as exc:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logger.debug("Unable to remove temp upload file after failure: %s", temp_path, exc_info=True)
            _update_ingest_job(
                job_id,
                status="failed",
                message=f"Upload failed: {str(exc)}",
            )
            raise HTTPException(status_code=400, detail="Failed to read upload stream.") from exc

        _update_ingest_job(
            job_id,
            status="queued",
            upload_bytes_received=bytes_received,
            upload_bytes_total=content_length if content_length is not None else bytes_received,
            message=f"Upload complete ({bytes_received} bytes). Queueing ingest job.",
        )
        _start_ingest_job_worker(
            job_id=job_id,
            temp_path=temp_path,
            run_id=resolved_run,
            customer_id=resolved_customer,
            column_mapping=column_mapping,
        )
        with ingest_jobs_lock:
            job = dict(ingest_jobs[job_id])
        return _public_ingest_job(job)

    @app.get("/ingest/jobs/{job_id}", response_model=IngestJobEnvelope)
    def get_ingest_job(
        job_id: str,
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        with ingest_jobs_lock:
            job = ingest_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
            if resolve_customer_id(job.get("customer_id")) != resolved_customer:
                raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
            return _public_ingest_job(job)

    @app.post("/integrations/pull/start", response_model=PullIntegrationStatusEnvelope)
    def start_pull_integration(
        payload: PullIntegrationStartRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        cfg_override = getattr(app.state, "integration_config_override", None)
        if isinstance(cfg_override, dict):
            cfg_doc = cfg_override
        else:
            path_override = getattr(app.state, "integration_config_path_override", None)
            path = str(path_override or "").strip() or os.getenv("NERAIUM_INTEGRATION_CONFIG_PATH")
            cfg_doc = load_integration_config(path)
        resolved_cfg = resolve_customer_integration(
            customer_id=resolved_customer,
            config_doc=cfg_doc,
        )

        endpoint_url = _validate_endpoint_url(
            payload.endpoint_url
            or str(resolved_cfg.get("endpoint_url") or "")
        )
        auth_type = str(payload.auth_type or resolved_cfg.get("auth_type") or "none")
        username = payload.username if payload.username is not None else resolved_cfg.get("username")
        password = payload.password if payload.password is not None else resolved_cfg.get("password")
        token = payload.token if payload.token is not None else resolved_cfg.get("token")
        polling_interval_seconds = _parse_finite_float(
            payload.polling_interval_seconds
            if payload.polling_interval_seconds is not None
            else resolved_cfg.get("polling_interval_seconds") or 30.0,
            field_name="polling_interval_seconds",
        )
        retry_max_attempts = _parse_int(
            payload.retry_max_attempts
            if payload.retry_max_attempts is not None
            else resolved_cfg.get("retry_max_attempts") or 3,
            field_name="retry_max_attempts",
        )
        retry_backoff_seconds = _parse_finite_float(
            payload.retry_backoff_seconds
            if payload.retry_backoff_seconds is not None
            else resolved_cfg.get("retry_backoff_seconds") or 1.0,
            field_name="retry_backoff_seconds",
        )
        request_timeout_seconds = _parse_finite_float(
            payload.request_timeout_seconds
            if payload.request_timeout_seconds is not None
            else resolved_cfg.get("request_timeout_seconds") or 10.0,
            field_name="request_timeout_seconds",
        )
        if polling_interval_seconds < 0.2:
            raise HTTPException(status_code=400, detail="polling_interval_seconds must be >= 0.2.")
        if retry_max_attempts < 1:
            raise HTTPException(status_code=400, detail="retry_max_attempts must be >= 1.")
        if retry_backoff_seconds < 0.05:
            raise HTTPException(status_code=400, detail="retry_backoff_seconds must be >= 0.05.")
        if request_timeout_seconds < 1.0:
            raise HTTPException(status_code=400, detail="request_timeout_seconds must be >= 1.0.")
        if auth_type == "basic":
            if not str(username or "").strip() or password is None:
                raise HTTPException(
                    status_code=400,
                    detail="Basic auth requires username and password.",
                )
        if auth_type == "bearer" and not str(token or "").strip():
            raise HTTPException(status_code=400, detail="Bearer auth requires token.")

        resolved_run = resolve_run_id_with_default(
            service_instance,
            payload.run_id,
            customer_id=resolved_customer,
            )
        _stop_pull_integration(resolved_customer, reason="Restarting integration.")
        stop_event = threading.Event()
        started_at = _utc_now_iso()
        with pull_integrations_lock:
            pull_integrations[resolved_customer] = {
                "customer_id": resolved_customer,
                "endpoint_url": endpoint_url,
                "run_id": resolved_run,
                "auth_type": auth_type,
                "username": username,
                "password": password,
                "token": token,
                "running": True,
                "status": "running",
                "polling_interval_seconds": polling_interval_seconds,
                "retry_max_attempts": retry_max_attempts,
                "retry_backoff_seconds": retry_backoff_seconds,
                "request_timeout_seconds": request_timeout_seconds,
                "started_at": started_at,
                "updated_at": started_at,
                "last_poll_at": None,
                "last_success_at": None,
                "last_error": None,
                "last_http_status": None,
                "total_polls": 0,
                "total_failures": 0,
                "consecutive_failures": 0,
                "total_ingested": 0,
                "message": "Pull integration started.",
                "_stop_event": stop_event,
                "_thread": None,
            }
            state = dict(pull_integrations[resolved_customer])
        _persist_pull_state(resolved_customer, state)
        _start_pull_worker(resolved_customer)
        log_structured(
            logger,
            event="pull_integration_started",
            fields={
                "customer_id": resolved_customer,
                "run_id": resolved_run,
                "endpoint_url": endpoint_url,
                "polling_interval_seconds": polling_interval_seconds,
                "auth_type": auth_type,
            },
            level=logging.INFO,
        )
        return _public_pull_state(state, customer_id=resolved_customer)

    @app.post("/integrations/pull/stop", response_model=PullIntegrationStatusEnvelope)
    def stop_pull_integration(
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        state = _stop_pull_integration(resolved_customer, reason="Pull integration stopped by operator.")
        log_structured(
            logger,
            event="pull_integration_stopped",
            fields={"customer_id": resolved_customer},
            level=logging.INFO,
        )
        return state

    @app.get("/integrations/pull/status", response_model=PullIntegrationStatusEnvelope)
    def pull_integration_status(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        with pull_integrations_lock:
            state = pull_integrations.get(resolved_customer)
            return _public_pull_state(state, customer_id=resolved_customer)

    @app.post("/reset", response_model=ActionResponse)
    def reset(_: None = Depends(require_api_key)) -> dict[str, bool]:
        logger.info("reset endpoint called")
        service_instance.reset()
        return {"ok": True}

    @app.get("/results/latest", response_model=ResultsEnvelope)
    def get_latest(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
        compact: bool = Query(default=False),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        latest = service_instance.get_latest_result(
            run_id=resolved,
            customer_id=resolved_customer,
            site_id=site_id,
        )
        latest_out = _compact_result_view(latest) if compact else latest
        results = [latest_out] if latest_out is not None else []
        return _results_envelope(results, latest=latest_out)

    @app.get("/results/recent", response_model=ResultsEnvelope)
    def get_recent(
        limit: int = 100,
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
        compact: bool = Query(default=False),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        results = service_instance.list_recent_results(
            limit=limit,
            run_id=resolved,
            customer_id=resolved_customer,
            site_id=site_id,
        )
        if compact:
            results = [_compact_result_view(r) for r in results]
        latest = results[0] if results else None
        return _results_envelope(results, latest=latest)

    @app.get("/results/export", response_model=ExportEnvelope)
    def export_results(
        format: Literal["json", "csv"] = Query(default="json"),
        limit: int = Query(default=500, ge=1, le=5000),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        results = service_instance.list_recent_results(
            limit=limit,
            run_id=resolved,
            customer_id=resolved_customer,
            site_id=site_id,
        )
        content_type, content = build_export(results, format_name=format)
        suffix = "json" if format == "json" else "csv"
        file_id = f"{resolved_customer}_{resolved or 'all_runs'}"
        filename = f"neraium_results_{file_id}.{suffix}"
        return {
            "run_id": resolved,
            "format": format,
            "count": len(results),
            "content_type": content_type,
            "filename": filename,
            "content": content,
        }

    @app.get("/results/export/download")
    def export_results_download(
        format: Literal["json", "csv"] = Query(default="json"),
        limit: int = Query(default=500, ge=1, le=5000),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
    ) -> StreamingResponse:
        env = export_results(
            format=format,
            limit=limit,
            run_id=run_id,
            customer_id=customer_id,
            site_id=site_id,
        )
        filename = str(env.get("filename") or f"neraium_results.{format}")
        content = str(env.get("content") or "")
        media_type = str(env.get("content_type") or "text/plain; charset=utf-8")
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(iter([content.encode("utf-8")]), media_type=media_type, headers=headers)

    @app.get("/results/{result_id}", response_model=ResultEnvelope)
    def get_result_by_id(
        result_id: int,
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
        resolved = resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        result = service_instance.get_result_by_id(
            result_id,
            run_id=resolved,
            customer_id=resolved_customer,
            )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        return {"result": result}

    @app.get("/export", response_model=ExportEnvelope)
    def export_results_legacy(
        format: Literal["json", "csv"] = Query(default="json"),
        limit: int = Query(default=500, ge=1, le=5000),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        return export_results(
            format=format,
            limit=limit,
            run_id=run_id,
            customer_id=customer_id,
            site_id=site_id,
        )

    app.include_router(build_web_router())
    _mount_web_static(app)

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
