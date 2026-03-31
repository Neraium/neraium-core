from __future__ import annotations

import base64
import json
import logging
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

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.types import Message

from .bootstrap.config import (
    cors_allow_headers,
    cors_allow_origin_regex,
    cors_allow_origins,
    dir_is_writable,
    request_body_limit_bytes,
    resolve_db_path,
    uvicorn_h11_max_incomplete_event_size,
)
from .bootstrap.errors import register_exception_handlers
from .bootstrap.logging import configure_logging
from .bootstrap.runtime import build_runtime_state_diagnostics, validate_runtime_or_raise
from .bootstrap.static import _mount_web_static
from .middleware.correlation import RequestCorrelationIdMiddleware
from .middleware.request_limits import MaxRequestBodySizeMiddleware
from .schemas.alerts import AlertAcknowledgeRequest, AlertResolveRequest, AlertsEnvelope
from .schemas.assistant import AssistantRequest, AssistantResponse, ReportRequest, ReportResponse
from .schemas.common import (
    ActionResponse,
    CanonicalOutputResponse,
    ClientErrorReport,
    CurrentStateEnvelope,
    DecisionEnvelope,
    EventsEnvelope,
    ExplanationEnvelope,
    ExportEnvelope,
    GeometryEnvelope,
    HealthResponse,
    HistoryEnvelope,
    RecommendationEnvelope,
    ResultEnvelope,
    ResultsEnvelope,
)
from .schemas.ingest import (
    BatchIngestRequest,
    CanonicalIngestRequest,
    CsvColumnMappingPayload,
    CsvIngestRequest,
    CsvPreviewRequest,
    CsvPreviewResponse,
    DemoCmapssStartRequest,
    DemoSeedRequest,
    IngestFrameRequest,
    IngestJobEnvelope,
    IngestRequest,
    JsonIngestRequest,
)
from .schemas.integrations import PullIntegrationStartRequest, PullIntegrationStatusEnvelope
from .schemas.runs import ActivateRunRequest, CreateRunRequest, LockBaselineRequest, RunEnvelope, RunsEnvelope, UpdateRunRequest

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
from .routers.ingest import build_ingest_router
from .routers.demo import build_demo_router
from .routers.integrations import build_integrations_router
from .routers.onboarding import build_onboarding_router
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
    get_core_runtime_status,
    log_structured,
    summarize_exception_for_logs,
)
from neraium_core.ingestion_normalization import (
    normalize_canonical_records_payload,
    normalize_external_batch_payload,
    normalize_external_payload,
)
from neraium_core.pipeline import infer_csv_mapping_stage, issue_to_dict, parse_csv_rows, validate_csv_mapping_stage


logger = logging.getLogger(__name__)


DEFAULT_UPLOAD_STREAM_CHUNK_BYTES = 1024 * 1024
DEFAULT_INGEST_JOB_MAX_ERROR_SAMPLES = 25

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


def create_app(
    service: StructuralMonitoringService | None = None,
    *,
    max_request_body_bytes: int | None = None,
) -> FastAPI:
    api_key = os.getenv("NERAIUM_API_KEY")
    configured_db_path = os.getenv("NERAIUM_DB_PATH", "neraium.db")
    db_path, persistence_available = resolve_db_path(configured_db_path)
    request_body_limit = (
        int(max_request_body_bytes)
        if max_request_body_bytes is not None
        else request_body_limit_bytes()
    )

    runtime_status = get_core_runtime_status()
    validate_runtime_or_raise(runtime_status)

    app = FastAPI(title="Neraium SII API", version="0.1.0")
    register_exception_handlers(app, logger=logger)
    app.add_middleware(RequestCorrelationIdMiddleware)
    app.add_middleware(MaxRequestBodySizeMiddleware, max_body_size=request_body_limit)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
    cors_allow_origins_list = cors_allow_origins()
    cors_allow_origin_regex_value = cors_allow_origin_regex()
    if cors_allow_origins_list or cors_allow_origin_regex_value:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_allow_origins_list,
            allow_origin_regex=cors_allow_origin_regex_value,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=cors_allow_headers(),
        )
        log_structured(
            logger,
            event="cors_configured",
            fields={
                "allow_origins_count": len(cors_allow_origins_list),
                "allow_origin_regex": bool(cors_allow_origin_regex_value),
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
    runtime_state_diagnostics: dict[str, Any] = build_runtime_state_diagnostics(
        request_body_limit=request_body_limit,
        db_path=db_path,
        writable_checker=dir_is_writable,
    )
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
    if not runtime_state_diagnostics.get("temp_dir_writable"):
        log_structured(
            logger,
            event="startup_temp_dir_unwritable",
            fields={"temp_dir": runtime_state_diagnostics.get("temp_dir")},
            level=logging.ERROR,
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
        status = str(job.get("status", "unknown"))
        ui_state = str(job.get("ui_state") or "")
        if not ui_state:
            if status == "uploading":
                ui_state = "uploading"
            elif status in {"queued", "processing"}:
                ui_state = "ingesting"
            elif status in {"completed", "partial_success", "failed"}:
                ui_state = status
            else:
                ui_state = "idle"
        return {
            "job_id": str(job.get("job_id")),
            "status": status,
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
            "lifecycle_phase": job.get("lifecycle_phase"),
            "ui_state": ui_state,
            "terminal_state": job.get("terminal_state"),
            "failure_category": job.get("failure_category"),
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
            current_status = str(job.get("status") or "")
            next_status = str(fields.get("status") or current_status)
            terminal_statuses = {"completed", "partial_success", "failed"}
            if current_status in terminal_statuses and next_status not in terminal_statuses:
                return dict(job)
            job.update(fields)
            job["updated_at"] = _utc_now_iso()
            if "partial_success" not in fields:
                job["partial_success"] = (
                    int(job.get("rows_succeeded", 0)) > 0 and int(job.get("rows_failed", 0)) > 0
                )
            status_value = str(job.get("status") or "")
            if status_value in {"uploading"}:
                job["lifecycle_phase"] = "uploading"
                job["ui_state"] = "uploading"
                job["terminal_state"] = None
                job["failure_category"] = None
            elif status_value in {"queued"}:
                job["lifecycle_phase"] = "queued"
                job["ui_state"] = "ingesting"
                job["terminal_state"] = None
                job["failure_category"] = None
            elif status_value in {"processing"}:
                job["lifecycle_phase"] = "processing"
                job["ui_state"] = "ingesting"
                job["terminal_state"] = None
                job["failure_category"] = None
            elif status_value in {"completed", "partial_success", "failed"}:
                job["lifecycle_phase"] = "terminal"
                job["ui_state"] = status_value
                job["terminal_state"] = status_value
                if status_value == "failed":
                    job["failure_category"] = str(job.get("failure_category") or "ingest_failed")
                else:
                    job["failure_category"] = None
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
        normalized, _ = normalize_external_batch_payload(rows, customer_id=customer_id)
        results = service_instance.ingest_normalized_frames(normalized, run_id=run_id, customer_id=customer_id)
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
            log_structured(
                logger,
                event="ingest_job_started",
                fields={
                    "job_id": job_id,
                    "run_id": run_id,
                    "customer_id": customer_id,
                    "ingest_path": "csv_upload",
                    "stage": "ingesting",
                },
                level=logging.INFO,
            )
            try:
                def _on_progress(progress: dict[str, Any]) -> None:
                    progress_status = str(progress.get("status") or "processing")
                    mapped_status = "processing"
                    if progress_status not in {"processing", "completed"}:
                        mapped_status = "processing"
                    _update_ingest_job(
                        job_id,
                        status=mapped_status,
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
                        "ingest_path": "csv_upload",
                        "stage": final_status,
                    },
                    level=logging.INFO,
                )
            except Exception as exc:
                message = actionable_validation_detail(str(exc))
                _update_ingest_job(job_id, status="failed", message=message, failure_category="ingest_failed")
                log_structured(
                    logger,
                    event="ingest_job_failed",
                    fields={
                        "job_id": job_id,
                        "run_id": run_id,
                        "customer_id": customer_id,
                        "ingest_path": "csv_upload",
                        "stage": "failed",
                        "error_type": "ingest_failed",
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
    app.include_router(
        build_ingest_router(
            service_instance=service_instance,
            require_api_key=require_api_key,
            resolve_customer_id=resolve_customer_id,
            resolve_run_id_with_default=resolve_run_id_with_default,
            actionable_validation_detail=actionable_validation_detail,
            normalize_external_payload=normalize_external_payload,
            normalize_external_batch_payload=normalize_external_batch_payload,
            normalize_canonical_records_payload=normalize_canonical_records_payload,
            process_alerts_after_ingest=_process_alerts_after_ingest,
            results_envelope=_results_envelope,
            parse_csv_rows=parse_csv_rows,
            infer_csv_mapping_stage=infer_csv_mapping_stage,
            validate_csv_mapping_stage=validate_csv_mapping_stage,
            issue_to_dict=issue_to_dict,
            request_body_limit=request_body_limit,
            normalize_content_length=_normalize_content_length,
            stream_upload_to_tempfile=_stream_upload_to_tempfile,
            update_ingest_job=_update_ingest_job,
            public_ingest_job=_public_ingest_job,
            persist_operational_state=_persist_operational_state,
            start_ingest_job_worker=_start_ingest_job_worker,
            ingest_jobs=ingest_jobs,
            ingest_jobs_lock=ingest_jobs_lock,
            models=type("IngestModels", (), {
                "ResultsEnvelope": ResultsEnvelope,
                "IngestRequest": IngestRequest,
                "IngestFrameRequest": IngestFrameRequest,
                "BatchIngestRequest": BatchIngestRequest,
                "CanonicalOutputResponse": CanonicalOutputResponse,
                "CsvIngestRequest": CsvIngestRequest,
                "CsvPreviewRequest": CsvPreviewRequest,
                "CsvPreviewResponse": CsvPreviewResponse,
                "IngestJobEnvelope": IngestJobEnvelope,
                "CsvColumnMappingPayload": CsvColumnMappingPayload,
                "utc_now_iso": _utc_now_iso,
            }),
        )
    )
    app.include_router(
        build_demo_router(
            service_instance=service_instance,
            require_api_key=require_api_key,
            resolve_customer_id=resolve_customer_id,
            resolve_run_id_with_default=resolve_run_id_with_default,
            run_demo_seed_job=_run_demo_seed_job,
            public_demo_job=_public_demo_job,
            demo_jobs=demo_jobs,
            demo_jobs_lock=demo_jobs_lock,
            load_cmapss_fd004_subset=_load_cmapss_fd004_subset,
            log_structured=log_structured,
            summarize_exception_for_logs=summarize_exception_for_logs,
            models=type("DemoModels", (), {"DemoSeedRequest": DemoSeedRequest, "DemoCmapssStartRequest": DemoCmapssStartRequest}),
            utc_now_iso=_utc_now_iso,
        )
    )
    app.include_router(
        build_integrations_router(
            app=app,
            require_api_key=require_api_key,
            resolve_customer_id=resolve_customer_id,
            resolve_run_id_with_default=resolve_run_id_with_default,
            service_instance=service_instance,
            stop_pull_integration=_stop_pull_integration,
            start_pull_worker=_start_pull_worker,
            public_pull_state=_public_pull_state,
            pull_integrations=pull_integrations,
            pull_integrations_lock=pull_integrations_lock,
            persist_pull_state=_persist_pull_state,
            validate_endpoint_url=_validate_endpoint_url,
            parse_finite_float=_parse_finite_float,
            parse_int=_parse_int,
            utc_now_iso=_utc_now_iso,
            log_structured=log_structured,
            models=type("IntegrationModels", (), {"PullIntegrationStartRequest": PullIntegrationStartRequest, "PullIntegrationStatusEnvelope": PullIntegrationStatusEnvelope}),
        )
    )

    app.include_router(
        build_onboarding_router(
            resolve_customer_id=resolve_customer_id,
            service_instance=service_instance,
            normalize_external_payload=normalize_external_payload,
            is_api_key_valid=is_api_key_valid,
            configured_api_key=api_key,
            ensure_default_run=_ensure_default_run,
            persist_operational_state=_persist_operational_state,
            store_instance=store_instance,
            persisted_state_enabled=persisted_state_enabled,
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


configure_logging()
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        h11_max_incomplete_event_size=uvicorn_h11_max_incomplete_event_size(),
    )
