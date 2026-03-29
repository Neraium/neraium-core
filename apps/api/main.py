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
from urllib.parse import urlparse
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
        if ext in {".html"}:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        elif ext in {".js", ".mjs", ".css", ".csv", ".txt", ".json", ".map"}:
            response.headers["Cache-Control"] = "public, max-age=86400, stale-while-revalidate=3600"
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
    customer_id: str | None = None
    timestamp: str | None = None
    site_id: str | None = None
    asset_id: str | None = None
    sensor_values: dict[str, Any] = Field(default_factory=dict)


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
        if isinstance(data, dict) and "items" not in data and "records" in data:
            remapped = dict(data)
            remapped["items"] = remapped.pop("records")
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
    try:
        instability = float(os.getenv("NERAIUM_ALERT_INSTABILITY_THRESHOLD", "1.5"))
    except (TypeError, ValueError):
        instability = 1.5
    try:
        drift_rapid = float(os.getenv("NERAIUM_ALERT_RAPID_DRIFT_DELTA", "0.2"))
    except (TypeError, ValueError):
        drift_rapid = 0.2
    return max(0.0, instability), max(0.0, drift_rapid)


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(n):
        return "-"
    return f"{n:.{digits}f}"


def _normalize_risk_level(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"HIGH", "MEDIUM", "LOW"}:
        return text
    return "UNKNOWN"


def _alert_context(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "result_id": result.get("result_id"),
        "run_id": result.get("run_id"),
        "timestamp": result.get("timestamp") or result.get("persisted_at"),
        "state": result.get("state") or result.get("interpreted_state"),
        "risk_level": result.get("risk_level"),
        "structural_drift_score": _safe_float(result.get("structural_drift_score"), 0.0),
        "composite_instability": _safe_float(result.get("latest_instability"), 0.0),
        "alert_status": result.get("alert_status") if isinstance(result.get("alert_status"), dict) else {},
    }


def _evaluate_alerts(
    *,
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    instability_threshold: float,
    rapid_drift_delta: float,
) -> list[dict[str, Any]]:
    del instability_threshold, rapid_drift_delta
    if not isinstance(current, dict):
        return []

    now = _utc_now_iso()
    current_status = current.get("alert_status") if isinstance(current.get("alert_status"), dict) else {}
    previous_status = previous.get("alert_status") if isinstance(previous, dict) and isinstance(previous.get("alert_status"), dict) else {}

    current_state = str(current_status.get("alert_state", "CLEAR")).upper()
    previous_state = str(previous_status.get("alert_state", "CLEAR")).upper()

    alerts: list[dict[str, Any]] = []
    should_emit_activation = current_state in {"ACTIVE_UNACKNOWLEDGED", "ESCALATED"} and previous_state not in {"ACTIVE_UNACKNOWLEDGED", "ACTIVE_ACKNOWLEDGED", "ESCALATED"}
    should_emit_renotify = bool(current_status.get("renotify_due")) and current_state in {"ACTIVE_UNACKNOWLEDGED", "ESCALATED"}

    if should_emit_activation or should_emit_renotify:
        severity = "critical" if current_state == "ESCALATED" else "high"
        if should_emit_activation:
            alert_type = "persistent_alert_activated"
            message = f"Persistent alert activated after {int(current_status.get('hit_window_threshold', 3))} consecutive hits."
        else:
            alert_type = "persistent_alert_renotify"
            message = "Persistent alert remains active and unacknowledged."

        alerts.append(
            {
                "id": f"alert_{uuid4().hex[:12]}",
                "type": alert_type,
                "severity": severity,
                "message": message,
                "created_at": now,
                "trigger": {
                    "state": current_state,
                    "reason": current_status.get("alert_reason"),
                    "consecutive_hit_count": int(current_status.get("consecutive_hit_count", 0)),
                    "hit_window_threshold": int(current_status.get("hit_window_threshold", 3)),
                    "unacknowledged_duration": int(current_status.get("unacknowledged_duration", 0)),
                },
                "context": _alert_context(current),
            }
        )

    return alerts



def _dispatch_alert_stubs(
    *,
    alert: dict[str, Any],
    webhook_url: str | None,
    email_to: str | None,
) -> None:
    if webhook_url:
        log_structured(
            logger,
            event="alert_webhook_stub",
            fields={
                "webhook_url": webhook_url,
                "alert_id": alert.get("id"),
                "alert_type": alert.get("type"),
                "severity": alert.get("severity"),
            },
            level=logging.INFO,
        )
    if email_to:
        log_structured(
            logger,
            event="alert_email_stub",
            fields={
                "email_to": email_to,
                "alert_id": alert.get("id"),
                "alert_type": alert.get("type"),
                "severity": alert.get("severity"),
            },
            level=logging.INFO,
        )


def _ensure_default_run(
    service: StructuralMonitoringService,
    *,
    customer_id: str | None,
) -> dict[str, Any]:
    resolved_customer = _resolve_customer_id(customer_id)
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


def _resolve_customer_id(customer_id: str | None) -> str:
    text = str(customer_id or "").strip()
    return text or "default-customer"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _actionable_validation_detail(message: str) -> str:
    text = str(message or "").strip()
    if not text:
        return "Validation failed. Check required fields and payload structure."
    if text.startswith("Invalid CSV row"):
        if "Invalid timestamp" in text:
            return (
                f"{text} Use ISO-8601 timestamps like "
                "2026-01-01T00:00:00+00:00."
            )
        if "Invalid signal value" in text or "Invalid signal type" in text:
            return f"{text} Ensure all sensor values are numeric or blank."
        return text
    if "Invalid timestamp" in text:
        return (
            "Invalid timestamp format. Use ISO-8601 timestamps like "
            "2026-01-01T00:00:00+00:00."
        )
    if "CSV must include" in text or "missing required columns" in text:
        return (
            f"{text} Ensure CSV header includes timestamp, site_id, asset_id "
            "plus at least one sensor column."
        )
    if "Could not infer" in text or "Provide a column_mapping" in text:
        return (
            f"{text} Use POST /ingest/csv/preview with a sample of your file, "
            "then send column_mapping (timestamp, asset_id, optional site_id, sensor_columns) with ingest."
        )
    if "Mapping requires" in text or "not present in the CSV header" in text:
        return (
            f"{text} Open the upload mapping panel and assign time, asset/entity, "
            "optional site, and one or more numeric sensor columns."
        )
    if "Invalid signal value" in text or "Invalid signal type" in text:
        return (
            f"{text} Ensure all sensor values are numeric or blank."
        )
    return text


def _resolve_run_id(
    service: StructuralMonitoringService,
    run_id: str | None,
    *,
    customer_id: str | None,
) -> str | None:
    if run_id is not None and str(run_id).strip():
        return str(run_id).strip()
    active = service.get_active_run(customer_id=_resolve_customer_id(customer_id))
    if active is None:
        return None
    rid = active.get("run_id")
    if rid is None:
        return None
    return str(rid)


def _require_run_id(
    service: StructuralMonitoringService,
    run_id: str | None,
    *,
    customer_id: str | None,
) -> str:
    resolved = _resolve_run_id(service, run_id, customer_id=customer_id)
    if resolved is None:
        raise HTTPException(status_code=400, detail="No active run. Create or activate a run first.")
    return resolved


def _request_run_id_or_active(
    service: StructuralMonitoringService,
    run_id: str | None,
    *,
    customer_id: str | None,
) -> str | None:
    if run_id is None:
        return _resolve_run_id(service, None, customer_id=customer_id)
    text = str(run_id).strip()
    if not text:
        return _resolve_run_id(service, None, customer_id=customer_id)
    return text


def _resolve_run_id_with_default(
    service: StructuralMonitoringService,
    run_id: str | None,
    *,
    customer_id: str | None,
) -> str:
    resolved = _request_run_id_or_active(service, run_id, customer_id=customer_id)
    if resolved is not None:
        return resolved
    created = _ensure_default_run(service, customer_id=_resolve_customer_id(customer_id))
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
        "geometry_path_length",
        "geometry_local_velocity_norm",
        "geometry_local_acceleration_norm",
        "geometry_curvature",
        "geometry_directional_consistency",
        "geometry_angular_change",
        "geometry_path_smoothness",
        "state_space_statistics_local_volume",
        "state_space_statistics_local_density",
        "state_space_statistics_covariance_trace",
        "state_space_statistics_principal_direction_strength",
        "state_space_statistics_anisotropy",
        "state_space_statistics_state_contraction_score",
        "state_space_statistics_state_expansion_score",
        "state_graph_node_count",
        "state_graph_edge_count",
        "state_graph_branching_factor",
        "state_graph_transition_entropy",
        "state_graph_revisit_rate",
        "state_graph_path_commitment_score",
        "state_graph_graph_divergence_score",
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
        geom = row.get("geometry") if isinstance(row.get("geometry"), dict) else {}
        stats = row.get("state_space_statistics") if isinstance(row.get("state_space_statistics"), dict) else {}
        sgraph = row.get("state_graph") if isinstance(row.get("state_graph"), dict) else {}
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
            geom.get("path_length"),
            geom.get("local_velocity_norm"),
            geom.get("local_acceleration_norm"),
            geom.get("curvature"),
            geom.get("directional_consistency"),
            geom.get("angular_change"),
            geom.get("path_smoothness"),
            stats.get("local_volume"),
            stats.get("local_density"),
            stats.get("covariance_trace"),
            stats.get("principal_direction_strength"),
            stats.get("anisotropy"),
            stats.get("state_contraction_score"),
            stats.get("state_expansion_score"),
            sgraph.get("node_count"),
            sgraph.get("edge_count"),
            sgraph.get("branching_factor"),
            sgraph.get("transition_entropy"),
            sgraph.get("revisit_rate"),
            sgraph.get("path_commitment_score"),
            sgraph.get("graph_divergence_score"),
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


def _graph_analytics_payload(
    analytics_dict: dict[str, Any],
    *,
    feature_names: list[str],
) -> dict[str, Any] | None:
    """Summarize engine graph metrics for geometry API consumers (UI, integrations)."""
    g = analytics_dict.get("graph")
    cg = analytics_dict.get("causal_graph")
    if not isinstance(g, dict) and not isinstance(cg, dict):
        return None
    out: dict[str, Any] = {}
    if isinstance(g, dict):
        out["correlation_graph"] = {
            "mean_degree": round(_safe_float(g.get("mean_degree")), 6),
            "density": round(_safe_float(g.get("density")), 6),
            "clustering": round(_safe_float(g.get("clustering")), 6),
            "connectivity": round(_safe_float(g.get("connectivity")), 6),
            "mean_absolute_connectivity": round(_safe_float(g.get("mean_absolute_connectivity")), 6),
        }
    if isinstance(cg, dict):
        raw_ds = cg.get("dominant_sources")
        idx_list: list[int] = []
        if isinstance(raw_ds, list):
            for x in raw_ds:
                try:
                    idx_list.append(int(x))
                except (TypeError, ValueError):
                    continue
        labels: list[str] = []
        for i in idx_list:
            if 0 <= i < len(feature_names):
                labels.append(str(feature_names[i]))
        out["causal_graph"] = {
            "density": round(_safe_float(cg.get("density")), 6),
            "asymmetry": round(_safe_float(cg.get("asymmetry")), 6),
            "dominant_source_indices": idx_list,
            "dominant_source_labels": labels,
        }
    return out or None


def _system_state_payload(result: dict[str, Any], *, analytics_dict: dict[str, Any]) -> dict[str, Any]:
    """Regime and interpretation snapshot aligned with the same result as geometry."""
    structural_flag = result.get("structural_analysis_available")
    if structural_flag is None:
        structural_available = not bool(analytics_dict.get("relational_metrics_skipped", True))
    else:
        structural_available = bool(structural_flag)

    regime_mem = result.get("regime_memory_state")
    regime_mem_dict = regime_mem if isinstance(regime_mem, dict) else {}

    rs = analytics_dict.get("regime_signature")
    rs_dict = rs if isinstance(rs, dict) else {}
    nearest = rs_dict.get("nearest")
    nearest_dict = nearest if isinstance(nearest, dict) else {}

    out: dict[str, Any] = {
        "structural_analysis_available": structural_available,
        "phase": result.get("phase") or result.get("interpreted_state") or result.get("state"),
        "interpreted_state": result.get("interpreted_state") or result.get("state"),
        "regime_name": result.get("regime_name"),
        "confidence": round(
            _safe_float(
                result.get("confidence"),
                _safe_float(result.get("confidence_score"), 0.0),
            ),
            4,
        ),
        "regime_memory": {
            "regime_name": regime_mem_dict.get("regime_name"),
            "library_size": regime_mem_dict.get("library_size"),
            "baseline_count": regime_mem_dict.get("baseline_count"),
        },
        "regime_signature": {
            "assigned_name": rs_dict.get("assigned_name"),
            "library_size": rs_dict.get("library_size"),
        },
    }
    rd = result.get("regime_distance")
    if rd is not None:
        out["regime_distance"] = round(_safe_float(rd), 4)
    ndist = nearest_dict.get("distance")
    if ndist is not None:
        out["nearest_regime"] = {
            "name": nearest_dict.get("name"),
            "distance": round(_safe_float(ndist), 6),
        }
    return out


def _fallback_correlation_from_relationships(n: int) -> np.ndarray:
    """PSD correlation matrix when stored correlation_geometry is missing but sensor names exist.

    Uses moderate equicorrelation (valid for any n when rho is chosen in the PSD range).
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=float)
    if n == 1:
        return np.ones((1, 1), dtype=float)
    rho = min(0.28, 1.0 / max(float(n - 1), 1.0))
    c = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(c, 1.0)
    return c


def _block_pad_correlation_matrix(corr: np.ndarray, m_add: int) -> np.ndarray:
    """Append a PSD correlation block for extra sensors (uncrossed with the existing block).

    Visualization-only: keeps the original matrix as a principal block and adds a small
    equicorrelation cluster so positions and edges remain well-defined for added channels.
    """
    n0 = int(corr.shape[0])
    m = int(m_add)
    if m <= 0:
        return corr
    a = np.asarray(corr, dtype=float)
    b = _fallback_correlation_from_relationships(m)
    out = np.zeros((n0 + m, n0 + m), dtype=float)
    out[:n0, :n0] = a
    out[n0:, n0:] = b
    np.fill_diagonal(out, 1.0)
    return out


def _expand_geometry_sensor_names(
    core_names: list[str],
    result: dict[str, Any],
    *,
    min_nodes: int,
    max_nodes: int,
) -> list[str]:
    """Prefer core (correlation order) first, then other channels from the run catalog."""
    out = [str(x).strip() for x in core_names if str(x).strip()]
    seen = set(out)
    if len(out) >= min_nodes or len(out) >= max_nodes:
        return out[:max_nodes]
    rel = result.get("sensor_relationships")
    if isinstance(rel, list):
        for x in rel:
            if len(out) >= max_nodes:
                break
            sx = str(x).strip()
            if sx and sx not in seen:
                out.append(sx)
                seen.add(sx)
            if len(out) >= min_nodes:
                return out[:max_nodes]
    sv = result.get("sensor_values")
    if isinstance(sv, dict):
        for k in sv.keys():
            if len(out) >= max_nodes:
                break
            sx = str(k).strip()
            if sx and sx not in seen:
                out.append(sx)
                seen.add(sx)
            if len(out) >= min_nodes:
                break
    return out[:max_nodes]


# Coherent structural-flow layout on XZ + vertical lift on Y (single ring n<=12, dual ring 13..N).
STRUCTURAL_FLOW_PLANE_MAX_N = 24

# When the engine keeps a smaller correlation matrix (variance-gated sensors) but the run still
# lists more asset channels, pad to at least this many nodes so the structural view matches catalog breadth.
GEOMETRY_VISUAL_MIN_NODES = 6


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


def _vertical_lift_y_from_stress(
    stress01: float,
    *,
    drift_global: float,
    inst_global: float,
    lift_max: float = 0.46,
) -> float:
    """Map normalized stress + global drift/instability to vertical separation (Y)."""
    drift_g = float(np.clip(drift_global, 0.0, 1.0))
    inst_g = float(np.clip(inst_global, 0.0, 1.0))
    global_scale = 1.0 + 0.45 * drift_g + 0.4 * inst_g
    s = float(np.clip(stress01, 0.0, 1.0))
    sev = max(0.0, (s - 0.12) / 0.88) ** 1.12
    sev = min(1.0, sev)
    return float(lift_max * sev * global_scale)


# Matches _stress_state / UI "in range" (stable): stay on the shared plane (y=0).
_STABLE_STRESS_CEILING = 0.33


def _lift_y_structural_flow(
    i: int,
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
) -> float:
    """In-range (stable) sensors sit on y=0; watch/critical lift with severity."""
    s = float(stress_norm[i]) if i < len(stress_norm) else 0.0
    if s < _STABLE_STRESS_CEILING:
        return 0.0
    return _vertical_lift_y_from_stress(s, drift_global=drift_global, inst_global=inst_global)


def _triangle_center_plane_four(
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Four sensors: equilateral triangle + center point, all in XZ; Y = lift when out-of-range.

    Indices 0–2 sit on the triangle; index 3 is the hub. Fully symmetric and readable on mobile.
    """
    _ = corr_matrix
    r = 0.62
    out = np.zeros((4, 3), dtype=float)
    for k in range(3):
        theta = math.pi / 2.0 + float(k) * (2.0 * math.pi / 3.0)
        out[k, 0] = r * math.cos(theta)
        out[k, 2] = r * math.sin(theta)
        s = float(stress_norm[k]) if k < len(stress_norm) else 0.0
        out[k, 1] = _lift_y_structural_flow(k, stress_norm, drift_global=drift_global, inst_global=inst_global)
    out[3, 0] = 0.0
    out[3, 2] = 0.0
    out[3, 1] = _lift_y_structural_flow(3, stress_norm, drift_global=drift_global, inst_global=inst_global)
    return out


def _diamond_plane_positions_four(
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Backward-compatible name: triangle + center plane layout for four sensors."""
    return _triangle_center_plane_four(
        stress_norm,
        drift_global=drift_global,
        inst_global=inst_global,
        corr_matrix=corr_matrix,
    )


def _plane_ring_positions(
    n: int,
    stress_norm: np.ndarray,
    *,
    drift_global: float,
    inst_global: float,
    corr_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """XZ plane: symmetric single ring (n<=12), dual ring (n>12); Y = lift when out-of-range."""
    _ = corr_matrix
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    out = np.zeros((n, 3), dtype=float)

    def _lift(i: int) -> float:
        return _lift_y_structural_flow(i, stress_norm, drift_global=drift_global, inst_global=inst_global)

    if n == 2:
        # Orthogonal placement (90°) — symmetric, not a diameter line through the origin.
        r = 0.56
        angles = (math.pi / 4.0, 3.0 * math.pi / 4.0)
        for i in range(2):
            theta = angles[i]
            out[i, 0] = r * math.cos(theta)
            out[i, 2] = r * math.sin(theta)
            out[i, 1] = _lift(i)
        return out

    if n <= 12:
        r = 0.5 + 0.02 * float(min(n, 12))
        for i in range(n):
            theta = 2.0 * math.pi * float(i) / float(n) - math.pi / 2.0
            out[i, 0] = r * math.cos(theta)
            out[i, 2] = r * math.sin(theta)
            out[i, 1] = _lift(i)
        return out

    n_inner = n // 2
    n_outer = n - n_inner
    r_in = 0.44
    r_out = 0.74 + 0.006 * float(max(0, n - 14))
    for i in range(n_inner):
        theta = 2.0 * math.pi * float(i) / float(max(n_inner, 1)) - math.pi / 2.0
        out[i, 0] = r_in * math.cos(theta)
        out[i, 2] = r_in * math.sin(theta)
        out[i, 1] = _lift(i)
    phase_off = math.pi / float(max(n_outer * 2, 1))
    for j in range(n_outer):
        idx = n_inner + j
        theta = 2.0 * math.pi * float(j) / float(max(n_outer, 1)) - math.pi / 2.0 + phase_off
        out[idx, 0] = r_out * math.cos(theta)
        out[idx, 2] = r_out * math.sin(theta)
        out[idx, 1] = _lift(idx)
    return out


def _project_geometry_positions(
    corr_current: np.ndarray,
    *,
    node_stress: np.ndarray,
    corr_baseline: np.ndarray | None,
    metrics: dict[str, Any] | None = None,
) -> np.ndarray:
    n = int(corr_current.shape[0])
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    if n == 1:
        return np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    drift_g = _safe_float(metrics.get("structural_drift_score"), 0.0) if metrics else 0.0
    inst_g = _safe_float(metrics.get("composite_instability"), 0.0) if metrics else 0.0
    if n == 4 and len(node_stress) >= 4:
        return _diamond_plane_positions_four(
            node_stress,
            drift_global=drift_g,
            inst_global=inst_g,
            corr_matrix=corr_current,
        )
    if 2 <= n <= STRUCTURAL_FLOW_PLANE_MAX_N and len(node_stress) >= n:
        return _plane_ring_positions(
            n,
            node_stress,
            drift_global=drift_g,
            inst_global=inst_g,
            corr_matrix=corr_current,
        )

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

    pos = np.stack([axis_x, axis_y, axis_z], axis=1)
    return _inflate_spectral_embedding_if_degenerate(pos, corr_matrix=corr_current, stress_norm=node_stress)


def _inflate_spectral_embedding_if_degenerate(
    pos: np.ndarray,
    *,
    corr_matrix: np.ndarray,
    stress_norm: np.ndarray,
) -> np.ndarray:
    """When spectral embedding collapses to ~1D, spread points on a circle in the weak subspace."""
    _ = corr_matrix
    _ = stress_norm
    n = int(pos.shape[0])
    if n < 3:
        return pos
    centered = pos - np.mean(pos, axis=0, keepdims=True)
    c = np.cov(centered.T)
    if not np.all(np.isfinite(c)):
        return pos
    ev, vecs = np.linalg.eigh(c)
    ev = np.maximum(ev, 0.0)
    if ev[-1] <= 1e-14:
        return pos
    if float(ev[-2] / ev[-1]) > 0.05:
        return pos
    e2 = vecs[:, -2]
    e3 = vecs[:, -3]
    out = pos.copy()
    scale = 0.048
    for i in range(n):
        ang = 2.0 * math.pi * float(i) / float(n)
        delta = e2 * (scale * math.cos(ang)) + e3 * (scale * math.sin(ang))
        out[i, :] += delta
    return out


def _stress_state(value: float) -> str:
    if value >= 0.66:
        return "critical"
    if value >= 0.33:
        return "watch"
    return "stable"


def _build_geometry_nodes(
    *,
    feature_names: list[str],
    positions: np.ndarray,
    magnitude_norm: np.ndarray,
    stress_norm: np.ndarray,
    core_name_set: set[str] | None = None,
) -> list[dict[str, Any]]:
    n = min(len(feature_names), int(positions.shape[0]), int(magnitude_norm.shape[0]), int(stress_norm.shape[0]))
    out: list[dict[str, Any]] = []
    for idx in range(n):
        stress = float(stress_norm[idx])
        state = _stress_state(stress)
        nm = str(feature_names[idx]).strip()
        in_corr = True if core_name_set is None else nm in core_name_set
        out.append(
            {
                "id": nm,
                "label": nm,
                "position": {
                    "x": round(float(positions[idx, 0]), 6),
                    "y": round(float(positions[idx, 1]), 6),
                    "z": round(float(positions[idx, 2]), 6),
                },
                "magnitude": round(float(magnitude_norm[idx]), 6),
                "stress": round(stress, 6),
                "state": state,
                "unstable": state == "critical",
                "is_unstable": state == "critical",
                "in_range": state == "stable",
                "in_correlation_window": in_corr,
                "role": "signal",
            }
        )
    return out


def _build_geometry_edges(
    corr_matrix: np.ndarray,
    *,
    feature_names: list[str],
    baseline_ref: np.ndarray | None = None,
    limit: int = 240,
    full_connectivity_max_n: int = STRUCTURAL_FLOW_PLANE_MAX_N,
) -> list[dict[str, Any]]:
    n = int(corr_matrix.shape[0])
    out: list[dict[str, Any]] = []
    if n <= 1:
        return out
    use_full = n <= int(full_connectivity_max_n)
    threshold = 0.0
    if not use_full:
        upper_idx = np.triu_indices(n, k=1)
        upper_abs = np.abs(corr_matrix[upper_idx])
        threshold = float(np.clip(np.percentile(upper_abs, 72.0), 0.22, 0.78)) if upper_abs.size else 1.1

    for i in range(n):
        for j in range(i + 1, n):
            weight = float(corr_matrix[i, j])
            magnitude = abs(weight)
            if not use_full and magnitude < threshold:
                continue
            baseline_weight = float(baseline_ref[i, j]) if baseline_ref is not None else weight
            delta = weight - baseline_weight
            out.append(
                {
                    "source": str(feature_names[i]),
                    "target": str(feature_names[j]),
                    "weight": round(weight, 6),
                    "magnitude": round(magnitude, 6),
                    "delta": round(delta, 6),
                    "type": "positive" if weight >= 0.0 else "negative",
                }
            )
    out.sort(key=lambda e: float(e.get("magnitude", 0.0)), reverse=True)
    return out[: max(1, int(limit))]


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
        "method": "structural_flow_coherent_plane_or_spectral",
        "is_visualization_projection": True,
        "source": "engine correlation geometry + graph analytics",
        "layout": "coherent_plane_ring",
        "plane_axes": ["x", "z"],
        "vertical_axis": "y",
        "note": (
            f"2–{STRUCTURAL_FLOW_PLANE_MAX_N} sensors: deterministic layout on shared XZ (triangle+hub when "
            f"n=4, even ring when n≤12, dual ring when n>12); in-range sensors stay on the plane; "
            f"out-of-range lift on +Y. Larger counts use spectral projection. Visualization-only."
        ),
    }
    provenance = {
        "engine_fields": [
            "sensor_relationships",
            "experimental_analytics.correlation_geometry.current",
            "experimental_analytics.correlation_geometry.baseline",
            "experimental_analytics.signal_structural_importance",
            "experimental_analytics.graph",
            "experimental_analytics.causal_graph",
            "experimental_analytics.regime_signature",
            "regime_memory_state",
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

    geometry_fallback = False
    if corr_current is None:
        if len(feature_names) < 1:
            graph_analytics = _graph_analytics_payload(analytics_dict, feature_names=feature_names)
            system_state = _system_state_payload(result, analytics_dict=analytics_dict)
            return {
                "run_id": run_id or result.get("run_id"),
                "result_id": result_id,
                "timestamp": result.get("timestamp") or result.get("persisted_at"),
                "available": False,
                "reason": "No sensor relationships or correlation geometry available for this result.",
                "metrics": metrics,
                "nodes": [],
                "edges": [],
                "views": {},
                "summary": {},
                "projection": projection,
                "provenance": provenance,
                "graph_analytics": graph_analytics,
                "system_state": system_state,
            }
        corr_current = _fallback_correlation_from_relationships(len(feature_names))
        corr_baseline = None
        geometry_fallback = True

    n = int(corr_current.shape[0])
    if len(feature_names) < n:
        feature_names = feature_names + [f"signal_{i + 1}" for i in range(len(feature_names), n)]
    if len(feature_names) > n:
        feature_names = feature_names[:n]

    core_names_before_expand = list(feature_names)
    n_core_corr = n
    expanded_names = _expand_geometry_sensor_names(
        feature_names,
        result,
        min_nodes=GEOMETRY_VISUAL_MIN_NODES,
        max_nodes=STRUCTURAL_FLOW_PLANE_MAX_N,
    )
    visual_expansion_count = 0
    if len(expanded_names) > n:
        visual_expansion_count = len(expanded_names) - n
        corr_current = _block_pad_correlation_matrix(corr_current, visual_expansion_count)
        if corr_baseline is not None and corr_baseline.shape == (n, n):
            corr_baseline = _block_pad_correlation_matrix(corr_baseline, visual_expansion_count)
        feature_names = expanded_names
        n = int(corr_current.shape[0])

    core_name_set = {str(x).strip() for x in core_names_before_expand if str(x).strip()}

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

    if 2 <= n <= STRUCTURAL_FLOW_PLANE_MAX_N:
        perm = np.array(sorted(range(n), key=lambda i: str(feature_names[i]).lower()), dtype=int)
        feature_names = [feature_names[i] for i in perm]
        corr_current = corr_current[np.ix_(perm, perm)]
        if corr_baseline is not None:
            corr_baseline = corr_baseline[np.ix_(perm, perm)]
        stress_norm = stress_norm[perm]
        importance_norm = importance_norm[perm]

    positions = _project_geometry_positions(
        corr_current,
        node_stress=stress_norm,
        corr_baseline=corr_baseline,
        metrics=metrics,
    )

    nodes_current = _build_geometry_nodes(
        feature_names=feature_names,
        positions=positions,
        magnitude_norm=importance_norm,
        stress_norm=stress_norm,
        core_name_set=core_name_set,
    )
    edges_current = _build_geometry_edges(
        corr_current,
        feature_names=feature_names,
        baseline_ref=corr_baseline,
        limit=400,
    )

    corr_reference = corr_baseline if corr_baseline is not None else corr_current
    importance_reference = np.mean(np.abs(corr_reference - np.eye(n)), axis=1)
    importance_reference_norm = _normalize_vector(importance_reference)
    stress_reference_norm = importance_reference_norm.copy()
    positions_reference = _project_geometry_positions(
        corr_reference,
        node_stress=stress_reference_norm,
        corr_baseline=corr_reference,
        metrics=metrics,
    )
    nodes_baseline = _build_geometry_nodes(
        feature_names=feature_names,
        positions=positions_reference,
        magnitude_norm=importance_reference_norm,
        stress_norm=stress_reference_norm,
        core_name_set=core_name_set,
    )
    edges_baseline = _build_geometry_edges(
        corr_reference,
        feature_names=feature_names,
        baseline_ref=corr_reference,
        limit=240,
    )
    baseline_by_id = {str(n.get("id")): n for n in nodes_baseline}
    for node in nodes_current:
        node_id = str(node.get("id"))
        base_node = baseline_by_id.get(node_id) or {}
        node["position_current"] = dict(node.get("position") or {})
        node["position_baseline"] = dict(base_node.get("position") or node.get("position") or {})

    summary = {
        "critical_nodes_current": sum(1 for n in nodes_current if str(n.get("state")) == "critical"),
        "watch_nodes_current": sum(1 for n in nodes_current if str(n.get("state")) == "watch"),
        "unstable_nodes_current": sum(1 for n in nodes_current if bool(n.get("unstable"))),
        "changed_edges_current": sum(1 for e in edges_current if abs(_safe_float(e.get("delta"), 0.0)) >= 0.10),
        "in_range_nodes": sum(1 for n in nodes_current if bool(n.get("in_range"))),
        "out_of_range_nodes": sum(1 for n in nodes_current if not bool(n.get("in_range"))),
    }

    graph_analytics = _graph_analytics_payload(analytics_dict, feature_names=feature_names)
    system_state = _system_state_payload(result, analytics_dict=analytics_dict)

    projection_out = dict(projection)
    if n == 4:
        projection_out["layout"] = "triangle_center_plane_four_sensors"
        projection_out["plane_axes"] = ["x", "z"]
        projection_out["vertical_axis"] = "y"
        projection_out["method"] = "triangle_center_plane_four_with_vertical_lift"
        projection_out["ring_node_count"] = 4
    elif 2 <= n <= STRUCTURAL_FLOW_PLANE_MAX_N:
        projection_out["layout"] = "coherent_plane_ring"
        projection_out["plane_axes"] = ["x", "z"]
        projection_out["vertical_axis"] = "y"
        projection_out["method"] = (
            "regular_polygon_plane_ring_with_vertical_lift"
            if n <= 12
            else "dual_ring_plane_with_vertical_lift"
        )
        projection_out["ring_node_count"] = n
    else:
        projection_out["layout"] = "spectral_correlation_projection"
        projection_out["plane_axes"] = None
        projection_out["vertical_axis"] = None
        projection_out["method"] = "spectral_projection_from_engine_correlation_geometry"
        projection_out.pop("ring_node_count", None)
    if geometry_fallback:
        projection_out["geometry_fallback"] = True
        base_note = str(projection_out.get("note") or "")
        projection_out["note"] = (
            base_note
            + " Correlation matrix was synthesized from sensor names because stored "
            "correlation_geometry.current was unavailable; baseline correlation is not available."
        ).strip()

    projection_out["correlation_core_count"] = int(n_core_corr)
    if visual_expansion_count > 0:
        projection_out["visual_expansion"] = {
            "enabled": True,
            "added_sensor_count": int(visual_expansion_count),
            "note": (
                "Additional channels from this run catalog were included so the structural view "
                "shows broader asset coverage. They use a separate correlation cluster (no cross-correlation "
                "to the engine estimation window); see each node's in_correlation_window flag."
            ),
        }

    return {
        "run_id": run_id or result.get("run_id"),
        "result_id": result_id,
        "timestamp": result.get("timestamp") or result.get("persisted_at"),
        "available": True,
        "reason": None,
        "metrics": metrics,
        "nodes": nodes_current,
        "edges": edges_current,
        "views": {
            "current": {
                "label": "Current structure",
                "source": "experimental_analytics.correlation_geometry.current",
                "available": True,
                "nodes": nodes_current,
                "edges": edges_current,
            },
            "baseline": {
                "label": "Baseline structure",
                "source": (
                    "experimental_analytics.correlation_geometry.baseline"
                    if corr_baseline is not None
                    else "baseline_not_available_using_current_structure_as_reference"
                ),
                "available": corr_baseline is not None,
                "nodes": nodes_baseline,
                "edges": edges_baseline,
            },
        },
        "summary": summary,
        "projection": projection_out,
        "provenance": provenance,
        "graph_analytics": graph_analytics,
        "system_state": system_state,
    }


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
    alert_instability_threshold, alert_rapid_drift_delta = _alert_thresholds()
    alert_webhook_url = str(os.getenv("NERAIUM_ALERT_WEBHOOK_URL") or "").strip() or None
    alert_email_to = str(os.getenv("NERAIUM_ALERT_EMAIL_TO") or "").strip() or None

    def _record_alerts_for_customer(customer_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        resolved_customer = _resolve_customer_id(customer_id)
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
        for alert in created:
            _dispatch_alert_stubs(
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
        items = _evaluate_alerts(
            current=latest_result,
            previous=previous_result,
            instability_threshold=alert_instability_threshold,
            rapid_drift_delta=alert_rapid_drift_delta,
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
            "customer_id": _resolve_customer_id(job.get("customer_id")),
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
            return dict(job)

    def _public_demo_job(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": str(job.get("job_id")),
            "status": str(job.get("status", "unknown")),
            "run_id": str(job.get("run_id") or ""),
            "customer_id": _resolve_customer_id(job.get("customer_id")),
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
        resolved_customer = _resolve_customer_id(customer_id)
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
        if isinstance(stop_event, threading.Event):
            stop_event.set()
        if isinstance(thread, threading.Thread) and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with pull_integrations_lock:
            final_state = pull_integrations.get(resolved_customer)
            return _public_pull_state(final_state, customer_id=resolved_customer)

    def _start_pull_worker(customer_id: str) -> None:
        resolved_customer = _resolve_customer_id(customer_id)

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

        worker = threading.Thread(target=_worker, daemon=True, name=f"pull-integration-{resolved_customer}")
        with pull_integrations_lock:
            state = pull_integrations.get(resolved_customer)
            if state is None:
                return
            state["_thread"] = worker
        worker.start()

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
                message = _actionable_validation_detail(str(exc))
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

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if not is_api_key_valid(api_key, x_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        latest = service_instance.get_latest_result(
            customer_id=_resolve_customer_id(os.getenv("NERAIUM_DEFAULT_CUSTOMER_ID"))
        )
        return HealthResponse(
            status="ok" if persistence_available else "degraded",
            version=app.version,
            auth_configured=bool(api_key),
            persistence_available=persistence_available,
            latest_result_available=latest is not None,
        )

    @app.post("/client-errors", status_code=status.HTTP_204_NO_CONTENT)
    def report_client_error(report: ClientErrorReport) -> Response:
        """Receive client-side JS errors from the web UI (no API key; same-origin)."""
        msg = (report.message or "")[:1500]
        url = (report.url or "")[:1500]
        stack = (report.stack or "")[:4000]
        extra = (report.reason or report.source or "")[:500]
        logger.warning(
            "client_js_error url=%s msg=%s extra=%s stack_snip=%s",
            url,
            msg,
            extra,
            stack[:800].replace("\n", " "),
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)

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
                customer_id=_resolve_customer_id(customer_id),
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
                customer_id=_resolve_customer_id(customer_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run": run}

    @app.get("/runs", response_model=RunsEnvelope)
    def list_runs(
        limit: int = Query(50, ge=1, le=500),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id)
        runs = service_instance.list_runs(limit=limit, customer_id=resolved_customer)
        active = service_instance.get_active_run(customer_id=resolved_customer)
        return {"active_run": active, "count": len(runs), "runs": runs}

    @app.get("/runs/active", response_model=RunEnvelope)
    def get_active_run(customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = service_instance.get_active_run(customer_id=_resolve_customer_id(customer_id))
        if run is None:
            return {"run": None}
        return {"run": run}

    @app.get("/runs/{run_id}", response_model=RunEnvelope)
    def get_run(run_id: str, customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        run = service_instance.get_run(run_id, customer_id=_resolve_customer_id(customer_id))
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        return {"run": run}

    @app.get("/runs/{run_id}/geometry", response_model=GeometryEnvelope)
    def get_run_geometry(
        run_id: str,
        result_id: int | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id)
        run = service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")

        if result_id is not None:
            result = service_instance.get_result_by_id(
                result_id,
                run_id=run_id,
                customer_id=resolved_customer,
            )
            if result is None:
                raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        else:
            result = service_instance.get_latest_result(run_id=run_id, customer_id=resolved_customer)

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
                "graph_analytics": None,
                "system_state": None,
            }

        return _build_geometry_payload(result, run_id=run_id)

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
                customer_id=_resolve_customer_id(customer_id),
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
                customer_id=_resolve_customer_id(customer_id),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"run": run}

    @app.get("/runs/{run_id}/baseline", response_model=dict)
    def get_run_baseline(
        run_id: str,
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id)
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
        resolved_customer = _resolve_customer_id(customer_id)
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
        resolved_customer = _resolve_customer_id(customer_id)
        try:
            service_instance.lock_baseline_for_run(
                run_id, locked=payload.locked, customer_id=resolved_customer
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
            resolved_customer = _resolve_customer_id(customer_id or payload.customer_id)
            resolved = _resolve_run_id_with_default(
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
            raise HTTPException(status_code=400, detail=_actionable_validation_detail(str(e)))
        return _results_envelope([result], latest=result)

    @app.post("/ingest/frame", response_model=CanonicalOutputResponse)
    def ingest_frame(
        payload: IngestFrameRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        try:
            resolved_customer = _resolve_customer_id(customer_id or payload.customer_id)
            resolved_run = _resolve_run_id_with_default(
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
            raise HTTPException(status_code=400, detail=_actionable_validation_detail(str(exc)))

    @app.post("/demo/seed/start")
    def demo_seed_start(
        payload: DemoSeedRequest,
        _: None = Depends(require_api_key),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id or payload.customer_id)
        resolved_run = _resolve_run_id_with_default(
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
        resolved_customer = _resolve_customer_id(customer_id or request_payload.customer_id)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run = _resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run = _resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run = _resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run = _resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run = _resolve_run_id(service_instance, payload.run_id, customer_id=resolved_customer)
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
            resolved_customer = _resolve_customer_id(customer_id or payload_customer)
            resolved = _resolve_run_id_with_default(
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
            detail = _actionable_validation_detail(str(e))
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
            resolved_customer = _resolve_customer_id(customer_id or payload.customer_id)
            resolved = _resolve_run_id_with_default(
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
            raise HTTPException(status_code=400, detail=_actionable_validation_detail(str(e)))
        return _results_envelope(results, latest=results[-1] if results else None)

    @app.post("/ingest/csv/preview", response_model=CsvPreviewResponse)
    def ingest_csv_preview(
        payload: CsvPreviewRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Infer semantic column roles from an arbitrary CSV header + sample rows."""
        _ = _resolve_customer_id(customer_id)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved_run = _resolve_run_id_with_default(
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
        resolved_customer = _resolve_customer_id(customer_id)
        with ingest_jobs_lock:
            job = ingest_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
            if _resolve_customer_id(job.get("customer_id")) != resolved_customer:
                raise HTTPException(status_code=404, detail=f"Unknown ingest job: {job_id}")
            return _public_ingest_job(job)

    @app.post("/integrations/pull/start", response_model=PullIntegrationStatusEnvelope)
    def start_pull_integration(
        payload: PullIntegrationStartRequest,
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id)
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

        resolved_run = _resolve_run_id_with_default(
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
        resolved_customer = _resolve_customer_id(customer_id)
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
        resolved_customer = _resolve_customer_id(customer_id)
        with pull_integrations_lock:
            state = pull_integrations.get(resolved_customer)
            return _public_pull_state(state, customer_id=resolved_customer)

    @app.get("/alerts", response_model=AlertsEnvelope)
    def list_alerts(
        limit: int = Query(default=50, ge=1, le=500),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id)
        with alerts_lock:
            items = [dict(a) for a in alerts.get(resolved_customer, [])]
        if run_id:
            items = [a for a in items if str((a.get("context") or {}).get("run_id") or "") == str(run_id)]
        latest = service_instance.get_current_state(run_id=run_id, customer_id=resolved_customer)
        if isinstance(latest, dict) and isinstance(latest.get("alert_status"), dict):
            status_item = {
                "id": f"alert_state_{latest.get('cycle', 'latest')}",
                "type": "alert_state",
                "severity": "critical" if str(latest["alert_status"].get("alert_state", "")).upper() == "ESCALATED" else ("high" if latest["alert_status"].get("alert_active") else "info"),
                "message": latest["alert_status"].get("alert_summary"),
                "created_at": latest["alert_status"].get("last_evaluated_at") or latest.get("timestamp"),
                "trigger": latest["alert_status"],
                "context": _alert_context(latest),
                "customer_id": resolved_customer,
            }
            items = [status_item, *items]
        items = items[:limit]
        return {"count": len(items), "alerts": items}

    @app.post("/alerts/acknowledge", response_model=ActionResponse)
    def acknowledge_alert(
        payload: AlertAcknowledgeRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, bool]:
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run_id = _request_run_id_or_active(service_instance, payload.run_id, customer_id=resolved_customer)
        service_instance.acknowledge_alert(
            run_id=resolved_run_id,
            customer_id=resolved_customer,
            acknowledged_by=payload.acknowledged_by,
        )
        return {"ok": True}

    @app.post("/alerts/resolve", response_model=ActionResponse)
    def resolve_alert(
        payload: AlertResolveRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, bool]:
        resolved_customer = _resolve_customer_id(payload.customer_id)
        resolved_run_id = _request_run_id_or_active(service_instance, payload.run_id, customer_id=resolved_customer)
        service_instance.resolve_alert(
            run_id=resolved_run_id,
            customer_id=resolved_customer,
            resolved_by=payload.resolved_by,
        )
        return {"ok": True}

    @app.post("/alerts/test", response_model=ActionResponse)
    def emit_test_alert(
        _: None = Depends(require_api_key),
        customer_id: str | None = Query(default=None),
        run_id: str | None = Query(default=None),
    ) -> dict[str, bool]:
        resolved_customer = _resolve_customer_id(customer_id)
        now = _utc_now_iso()
        alert = {
            "id": f"alert_{uuid4().hex[:12]}",
            "type": "test_alert",
            "severity": "info",
            "message": "Test alert generated manually.",
            "created_at": now,
            "trigger": {"manual": True},
            "context": {
                "result_id": None,
                "run_id": run_id,
                "timestamp": now,
                "state": "TEST",
                "risk_level": "UNKNOWN",
                "structural_drift_score": 0.0,
                "composite_instability": 0.0,
            },
        }
        _record_alerts_for_customer(resolved_customer, [alert])
        return {"ok": True}

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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        results = service_instance.list_recent_results(
            limit=limit,
            run_id=resolved,
            customer_id=resolved_customer,
            site_id=site_id,
        )
        content_type, content = _build_export(results, format_name=format)
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
        resolved_customer = _resolve_customer_id(customer_id)
        resolved = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        result = service_instance.get_result_by_id(
            result_id,
            run_id=resolved,
            customer_id=resolved_customer,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        return {"result": result}

    @app.get("/results/{result_id}/geometry", response_model=GeometryEnvelope)
    def get_result_geometry(
        result_id: int,
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = _resolve_customer_id(customer_id)
        resolved = _resolve_run_id(service_instance, run_id, customer_id=resolved_customer)
        result = service_instance.get_result_by_id(
            result_id,
            run_id=resolved,
            customer_id=resolved_customer,
        )
        if result is None:
            raise HTTPException(status_code=404, detail=f"Unknown result_id: {result_id}")
        return _build_geometry_payload(result, run_id=result.get("run_id") or resolved)

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
