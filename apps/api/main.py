from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

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
from .bootstrap.runtime import (
    build_runtime_state_diagnostics,
    resolve_runtime_mode_from_env,
    validate_runtime_or_raise,
    validate_startup_assumptions_or_raise,
)
from .bootstrap.static import _mount_web_static
from .middleware.correlation import RequestCorrelationIdMiddleware
from .middleware.request_limits import MaxRequestBodySizeMiddleware
from .middleware.security_headers import SecurityHeadersMiddleware
from .schemas.alerts import AlertAcknowledgeRequest, AlertResolveRequest, AlertsEnvelope
from .schemas.assistant import AssistantRequest, AssistantResponse, ReportRequest, ReportResponse
from .schemas.chat import ChatRequest, ChatResponse
from .schemas.common import (
    ActionResponse,
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
from .schemas.runs import ActivateRunRequest, CreateRunRequest, LockBaselineRequest, RunEnvelope, RunsEnvelope, UpdateRunRequest

from .integration import (
    IntegrationMappingError,
    apply_integration_mapping,
    load_integration_config,
)
from .web import build_web_router
from .routers.health import build_health_router
from .routers.alerts import build_alerts_router
from .routers.geometry import build_geometry_router
from .routers.ingest import build_ingest_router
from .routers.demo import build_demo_router
from .routers.demo_playback import build_demo_playback_router
from .routers.integrations import build_integrations_router
from .routers.onboarding import build_onboarding_router
DEFAULT_MAX_REQUEST_BODY_BYTES = request_body_limit_bytes()

from .routers.dependencies import (
    DemoRouterDependencies,
    IngestRouterDependencies,
    IntegrationsRouterDependencies,
    OnboardingRouterDependencies,
)
from .services.alerts import alert_thresholds as service_alert_thresholds, evaluate_alerts, dispatch_alert_stubs
from .services.request_context import (
    resolve_customer_id,
    resolve_customer_id_for_run,
    resolve_run_id,
    request_run_id_or_active,
    resolve_run_id_with_default,
)
from .services.validation_utils import actionable_validation_detail
from .services.export_utils import build_export
from .services.state_store import RuntimeStateStore
from .services.operational_state import OperationalStateService
from .services.ingest_jobs import IngestJobsManager
from .services.demo_jobs import DemoJobsManager
from .services.pull_integrations import PullIntegrationsManager
from .services.state_restore import OperationalStateRestoreService
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
from neraium_core.runtime_config import RuntimeConfig
from neraium_core.pipeline import infer_csv_mapping_stage, issue_to_dict, parse_csv_rows, validate_csv_mapping_stage


logger = logging.getLogger(__name__)

_RECENT_RESULTS_TIMEOUT_S = 0.45
_RECENT_RESULTS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="recent-results")


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
    envelope = {"latest": latest, "count": len(results), "results": results}
    canonical = latest.get("canonical_output") if isinstance(latest, dict) and isinstance(latest.get("canonical_output"), dict) else {}
    if isinstance(canonical.get("alert_status"), dict):
        envelope["alert_status"] = canonical.get("alert_status")
    if isinstance(canonical.get("memory_recall"), dict):
        envelope["memory_recall"] = canonical.get("memory_recall")
    return envelope


def _compact_result_view(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return result
    compact: dict[str, Any] = {
        "timestamp": result.get("timestamp"),
        "site_id": result.get("site_id"),
        "asset_id": result.get("asset_id"),
        "state": result.get("state"),
        "regime_name": result.get("regime_name"),
        "risk_level": result.get("risk_level"),
        "raw_system_health": result.get("raw_system_health"),
        "display_health": result.get("display_health"),
        "system_health": result.get("system_health"),  # Backward-compat: same as display_health
        "structural_drift_score": result.get("structural_drift_score"),
        "alert": result.get("alert"),
        "confidence": result.get("confidence"),
        "run_id": result.get("run_id"),
    }
    return compact


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


def _chat_safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _chat_regime_name(latest: dict[str, Any]) -> str:
    session = latest.get("session") if isinstance(latest.get("session"), dict) else {}
    return str(
        latest.get("regime_name")
        or session.get("regime_name")
        or latest.get("state")
        or latest.get("interpreted_state")
        or "unknown"
    )


def _chat_system_health(latest: dict[str, Any]) -> float | None:
    risk = latest.get("risk_assessment") if isinstance(latest.get("risk_assessment"), dict) else {}
    risk_level = str(risk.get("risk_level") or latest.get("risk_level") or "UNKNOWN").upper()
    drift = _chat_safe_float(latest.get("structural_drift_score"))
    instability = _chat_safe_float(risk.get("latest_instability"))
    if instability is None:
        exp = latest.get("experimental_analytics") if isinstance(latest.get("experimental_analytics"), dict) else {}
        instability = _chat_safe_float(exp.get("composite_instability"))
    base = 90.0
    if risk_level == "HIGH":
        base -= 42.0
    elif risk_level == "MEDIUM":
        base -= 24.0
    elif risk_level not in {"LOW", "MEDIUM", "HIGH"}:
        base -= 10.0
    if drift is not None:
        base -= min(28.0, max(0.0, drift) * 34.0)
    if instability is not None:
        base -= min(26.0, max(0.0, instability) * 32.0)
    return max(0.0, min(100.0, round(base, 2)))


def create_app(
    service: StructuralMonitoringService | None = None,
    *,
    max_request_body_bytes: int | None = None,
) -> FastAPI:
    runtime_cfg = RuntimeConfig.from_env()
    runtime_mode = resolve_runtime_mode_from_env()
    api_key = os.getenv("NERAIUM_API_KEY")
    configured_db_path = os.getenv("NERAIUM_DB_PATH", "neraium.db")
    db_path, persistence_available = resolve_db_path(configured_db_path)
    request_body_limit = (
        int(max_request_body_bytes)
        if max_request_body_bytes is not None
        else DEFAULT_MAX_REQUEST_BODY_BYTES
    )

    runtime_status = get_core_runtime_status()
    validate_runtime_or_raise(runtime_status)

    log_structured(
        logger,
        event="startup_config_loaded",
        fields={
            **runtime_cfg.to_diagnostics(),
            "runtime_mode": runtime_mode,
            "db_path": db_path,
            "persistence_available": bool(persistence_available),
            "request_body_limit": int(request_body_limit),
        },
        level=logging.INFO,
    )

    app = FastAPI(title="Neraium SII API", version="0.1.0")
    app.state.request_body_limit = request_body_limit
    register_exception_handlers(app, logger=logger)
    app.add_middleware(RequestCorrelationIdMiddleware)
    app.add_middleware(MaxRequestBodySizeMiddleware, max_body_size=request_body_limit)
    app.add_middleware(SecurityHeadersMiddleware)
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
    service_instance = service or StructuralMonitoringService(
        store=ResultStore(db_path=db_path),
        baseline_window=runtime_cfg.baseline_window,
        recent_window=runtime_cfg.recent_window,
    )
    integration_config_path = os.getenv("NERAIUM_INTEGRATION_CONFIG_PATH")
    integration_config = load_integration_config(integration_config_path)
    app.state.integration_config_override = integration_config
    app.state.integration_config_path_override = integration_config_path
    state_store = RuntimeStateStore()
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
    runtime_state_diagnostics.update(runtime_cfg.to_diagnostics())
    runtime_state_diagnostics["runtime_mode"] = runtime_mode
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

    validate_startup_assumptions_or_raise(
        runtime_mode=runtime_mode,
        persistence_available=bool(persistence_available),
        runtime_state_diagnostics=runtime_state_diagnostics,
    )

    log_structured(
        logger,
        event="startup_validation_passed",
        fields={
            "runtime_mode": runtime_mode,
            "persisted_state_enabled": bool(persisted_state_enabled),
            "db_path_writable": bool(runtime_state_diagnostics.get("db_path_writable")),
            "temp_dir_writable": bool(runtime_state_diagnostics.get("temp_dir_writable")),
        },
        level=logging.INFO,
    )

    operational_state = OperationalStateService(store_instance=store_instance, enabled=persisted_state_enabled, logger=logger)

    def _persist_operational_state(key: str, payload: dict[str, Any], *, customer_id: str | None = None, run_id: str | None = None) -> None:
        operational_state.persist(key, payload, customer_id=customer_id, run_id=run_id)

    def _delete_operational_state(key: str) -> None:
        operational_state.delete(key)

    def _record_alerts_for_customer(customer_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not items:
            return []
        resolved_customer = resolve_customer_id(customer_id)
        created: list[dict[str, Any]] = []
        with state_store.alerts_lock:
            bucket = state_store.alerts.setdefault(resolved_customer, [])
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
        current_for_alerts = latest_result
        if isinstance(latest_result, dict) and isinstance(latest_result.get("canonical_output"), dict):
            current_for_alerts = latest_result["canonical_output"]
        previous_for_alerts = previous_result
        if isinstance(previous_result, dict) and isinstance(previous_result.get("canonical_output"), dict):
            previous_for_alerts = previous_result["canonical_output"]
        items = evaluate_alerts(
            current=current_for_alerts,
            previous=previous_for_alerts,
            instability_threshold=alert_instability_threshold,
            rapid_drift_delta=alert_rapid_drift_delta,
            now_iso=_utc_now_iso(),
        )
        if run_id:
            for item in items:
                context = item.get("context")
                if isinstance(context, dict):
                    if not context.get("run_id"):
                        context["run_id"] = run_id
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

    ingest_manager = IngestJobsManager(
        store=state_store,
        service_instance=service_instance,
        operational_state=operational_state,
        resolve_customer_id=resolve_customer_id,
        utc_now_iso=_utc_now_iso,
        process_alerts_after_ingest=_process_alerts_after_ingest,
        actionable_validation_detail=actionable_validation_detail,
        log_structured=log_structured,
        summarize_exception_for_logs=summarize_exception_for_logs,
        logger=logger,
    )
    demo_manager = DemoJobsManager(
        store=state_store,
        service_instance=service_instance,
        resolve_customer_id=resolve_customer_id,
        utc_now_iso=_utc_now_iso,
        log_structured=log_structured,
        summarize_exception_for_logs=summarize_exception_for_logs,
        logger=logger,
    )
    pull_manager = PullIntegrationsManager(
        app=app,
        store=state_store,
        service_instance=service_instance,
        operational_state=operational_state,
        resolve_customer_id=resolve_customer_id,
        utc_now_iso=_utc_now_iso,
        safe_float=_safe_float,
        parse_int=_parse_int,
        parse_finite_float=_parse_finite_float,
        log_structured=log_structured,
        summarize_exception_for_logs=summarize_exception_for_logs,
        load_integration_config=load_integration_config,
        apply_integration_mapping=apply_integration_mapping,
        integration_mapping_error=IntegrationMappingError,
        logger=logger,
    )
    OperationalStateRestoreService(
        operational_state=operational_state,
        store=state_store,
        resolve_customer_id=resolve_customer_id,
        utc_now_iso=_utc_now_iso,
        pull_manager=pull_manager,
        runtime_state_diagnostics=runtime_state_diagnostics,
        logger=logger,
        log_structured=log_structured,
    ).restore()

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
            alerts=state_store.alerts,
            alerts_lock=state_store.alerts_lock,
            record_alerts_for_customer=_record_alerts_for_customer,
            alerts_envelope_model=AlertsEnvelope,
            action_response_model=ActionResponse,
            alert_ack_model=AlertAcknowledgeRequest,
            alert_resolve_model=AlertResolveRequest,
        )
    )
    app.include_router(
        build_ingest_router(
            deps=IngestRouterDependencies(
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
                stream_upload_to_tempfile=ingest_manager.stream_upload_to_tempfile,
                update_ingest_job=ingest_manager.update_ingest_job,
                public_ingest_job=ingest_manager.public_ingest_job,
                persist_operational_state=_persist_operational_state,
                start_ingest_job_worker=ingest_manager.start_ingest_job_worker,
                ingest_jobs=state_store.ingest_jobs,
                ingest_jobs_lock=state_store.ingest_jobs_lock,
                utc_now_iso=_utc_now_iso,
            ),
        )
    )
    app.include_router(
        build_demo_router(
            deps=DemoRouterDependencies(
                service_instance=service_instance,
                require_api_key=require_api_key,
                resolve_customer_id=resolve_customer_id,
                resolve_run_id_with_default=resolve_run_id_with_default,
                start_demo_seed_job=demo_manager.start_demo_seed_job,
                public_demo_job=demo_manager.public_demo_job,
                demo_jobs=state_store.demo_jobs,
                demo_jobs_lock=state_store.demo_jobs_lock,
                load_greenhouse_demo_subset=demo_manager.load_greenhouse_demo_subset,
                log_structured=log_structured,
                summarize_exception_for_logs=summarize_exception_for_logs,
                utc_now_iso=_utc_now_iso,
            ),
        )
    )
    app.include_router(
        build_integrations_router(
            deps=IntegrationsRouterDependencies(
                app=app,
                require_api_key=require_api_key,
                resolve_customer_id=resolve_customer_id,
                resolve_run_id_with_default=resolve_run_id_with_default,
                service_instance=service_instance,
                pull_manager=pull_manager,
                log_structured=log_structured,
            ),
        )
    )

    app.include_router(build_demo_playback_router())

    app.include_router(
        build_onboarding_router(
            deps=OnboardingRouterDependencies(
                resolve_customer_id=resolve_customer_id,
                service_instance=service_instance,
                normalize_external_payload=normalize_external_payload,
                is_api_key_valid=is_api_key_valid,
                configured_api_key=api_key,
                ensure_default_run=_ensure_default_run,
                persist_operational_state=_persist_operational_state,
                store_instance=store_instance,
                persisted_state_enabled=persisted_state_enabled,
            ),
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



    @app.get(
        "/v2/state",
        response_model=CurrentStateEnvelope,
        openapi_extra={
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "example": {
                                "state": {"contract_version": "decision-contract.v2"}
                            }
                        }
                    }
                }
            }
        },
    )
    def get_state_v2(
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        response = get_state(run_id=run_id, customer_id=customer_id)
        state = dict(response.get("state") or {})
        state.setdefault("contract_version", "decision-contract.v2")
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

    @app.post("/api/chat", response_model=ChatResponse)
    def api_chat(
        payload: ChatRequest,
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        req_ctx = payload.context if payload.context is not None else None
        resolved_customer = resolve_customer_id(getattr(req_ctx, "customer_id", None))
        requested_run = getattr(req_ctx, "run_id", None)
        requested_site = str(getattr(req_ctx, "site_id", "") or "").strip() or None
        requested_asset = str(getattr(req_ctx, "asset_id", "") or "").strip() or None
        resolved_run = resolve_run_id(service_instance, requested_run, customer_id=resolved_customer)

        latest = service_instance.get_latest_result(
            run_id=resolved_run,
            customer_id=resolved_customer,
            site_id=requested_site,
        )
        history = service_instance.list_recent_results(
            limit=30,
            run_id=resolved_run,
            customer_id=resolved_customer,
            site_id=requested_site,
        )

        if isinstance(latest, dict) and requested_asset:
            latest_asset = str(latest.get("asset_id") or "").strip()
            if latest_asset and latest_asset != requested_asset:
                latest = next(
                    (row for row in history if isinstance(row, dict) and str(row.get("asset_id") or "").strip() == requested_asset),
                    latest,
                )

        if not isinstance(latest, dict):
            observed = ["No current snapshot is available for the requested run/site/asset context."]
            inferred = ["Interpretation confidence is low because telemetry context is incomplete."]
            suggested = "Select or activate the correct run context, ingest telemetry, then re-run interpretation."
            return {
                "observed": observed,
                "inferred": inferred,
                "suggested_next_step": suggested,
                "response_text": "\n".join([*observed, *inferred, suggested]),
                "uncertainty": 1.0,
                "grounding": {
                    "user_message": payload.message,
                    "context": (req_ctx.model_dump() if req_ctx is not None else {}),
                    "prompt": "No current state available; interpretation cannot be grounded yet.",
                    "metrics": {},
                },
            }

        risk = latest.get("risk_assessment") if isinstance(latest.get("risk_assessment"), dict) else {}
        recommendation = latest.get("operational_recommendation") if isinstance(latest.get("operational_recommendation"), dict) else {}
        rec = recommendation or {}
        session = latest.get("session") if isinstance(latest.get("session"), dict) else {}
        trend = str(risk.get("trend") or latest.get("trend") or "unknown")
        risk_level = str(risk.get("risk_level") or latest.get("risk_level") or "UNKNOWN")
        drift = _chat_safe_float(latest.get("structural_drift_score"))
        instability = _chat_safe_float(risk.get("latest_instability"))
        if instability is None:
            exp = latest.get("experimental_analytics") if isinstance(latest.get("experimental_analytics"), dict) else {}
            instability = _chat_safe_float(exp.get("composite_instability"))
        system_health = _chat_system_health(latest)
        regime_name = _chat_regime_name(latest)
        explanation_text = str(latest.get("explanation_text") or "No explanation text available.")
        sensor_relationships = latest.get("sensor_relationships")
        confidence_score = _chat_safe_float(latest.get("confidence"))

        history_rows = [row for row in history if isinstance(row, dict)]
        history_oldest_to_latest = list(reversed(history_rows))
        drift_values = [_chat_safe_float(row.get("structural_drift_score")) for row in history_oldest_to_latest]
        drift_values = [v for v in drift_values if v is not None]
        system_health_values = [_chat_system_health(row) for row in history_oldest_to_latest]
        system_health_values = [v for v in system_health_values if v is not None]
        what_changed = "Insufficient history to summarize recent change."
        if len(drift_values) >= 2 or len(system_health_values) >= 2:
            d_msg = "drift trend unavailable"
            if len(drift_values) >= 2:
                d_delta = drift_values[-1] - drift_values[0]
                d_msg = f"drift {'increased' if d_delta > 0.01 else 'decreased' if d_delta < -0.01 else 'remained stable'} ({d_delta:+.3f})"
            h_msg = "system health trend unavailable"
            if len(system_health_values) >= 2:
                h_delta = system_health_values[-1] - system_health_values[0]
                h_msg = f"system health {'improved' if h_delta > 0.8 else 'declined' if h_delta < -0.8 else 'was stable'} ({h_delta:+.2f})"
            what_changed = f"Recent window: {d_msg}; {h_msg}."

        uncertainty = 1.0
        if confidence_score is not None:
            uncertainty = max(0.0, min(1.0, 1.0 - confidence_score))
        elif system_health is not None:
            uncertainty = max(0.0, min(1.0, (100.0 - system_health) / 100.0))

        observed = [
            f"Context: regime={regime_name}, risk={risk_level}, trend={trend}.",
            f"Signals: drift={drift if drift is not None else 'not available'}, health={system_health if system_health is not None else 'not available'}, confidence={confidence_score if confidence_score is not None else 'not available'}.",
            f"Current explanation: {explanation_text}",
            what_changed,
        ]
        inferred = [
            (
                "Window interpretation: risk pressure is rising."
                if str(trend).lower() in {"increasing", "up", "rising"}
                else "Window interpretation: risk is stable or mixed."
            ),
            f"Uncertainty={round(float(uncertainty), 4)} (higher means lower certainty).",
        ]
        rec_text = str(rec.get("action") or rec.get("immediate_action") or rec.get("recommendation") or "").strip()
        normalized_risk = str(risk_level).upper()
        if normalized_risk == "HIGH":
            suggested_next_step = rec_text or "Execute the active recommendation now; validate highest-drift signals and alert status."
        elif normalized_risk == "MEDIUM":
            suggested_next_step = rec_text or "Apply the current recommendation and keep elevated monitoring cadence."
        else:
            suggested_next_step = "Maintain baseline monitoring cadence and confirm no new anomalies."

        provided_metrics = payload.recent_metrics_snapshot if isinstance(payload.recent_metrics_snapshot, dict) else {}
        provided_keys = sorted(str(k) for k in provided_metrics.keys())
        if provided_keys:
            observed.append(
                f"Caller metrics snapshot keys: {', '.join(provided_keys[:12])}."
            )
        if isinstance(provided_metrics.get("transition_count_recent"), (int, float)):
            inferred.append(
                f"transition_count_recent={int(provided_metrics['transition_count_recent'])} was included with stored run history."
            )

        return {
            "observed": observed,
            "inferred": inferred,
            "suggested_next_step": suggested_next_step,
            "response_text": "\n".join([*observed, *inferred, suggested_next_step]),
            "uncertainty": round(float(uncertainty), 4),
            "grounding": {
                "user_message": payload.message,
                "context": {
                    "customer_id": resolved_customer,
                    "run_id": resolved_run,
                    "site_id": requested_site or session.get("site_id"),
                    "asset_id": requested_asset or session.get("asset_id") or latest.get("asset_id"),
                },
                "prompt": "Grounded from current run state, recent history, and caller-provided metrics snapshot.",
                "metrics": {
                    "structural_drift_score": drift,
                    "system_health": system_health,
                    "regime_name": regime_name,
                    "explanation_text": explanation_text,
                    "sensor_relationships": sensor_relationships,
                    "confidence_score": confidence_score,
                    "recent_history_window": [
                        {
                            "timestamp": row.get("timestamp") or row.get("persisted_at"),
                            "structural_drift_score": _chat_safe_float(row.get("structural_drift_score")),
                            "system_health": _chat_system_health(row),
                            "risk_level": (row.get("risk_assessment") or {}).get("risk_level") if isinstance(row.get("risk_assessment"), dict) else row.get("risk_level"),
                        }
                        for row in history_rows[:20]
                    ],
                    "recent_metrics_snapshot": payload.recent_metrics_snapshot or {},
                },
                "guardrails": {
                    "no_hallucination": True,
                    "no_invented_data": True,
                    "explicit_uncertainty": True,
                },
            },
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
        resolved_customer = resolve_customer_id_for_run(service_instance, customer_id, run_id)
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
        started_at = time.perf_counter()
        requested_run = str(run_id).strip() if run_id is not None else ""
        logger.info("ENTER /results/recent run_id=%s", requested_run or None)

        resolved_customer = resolve_customer_id_for_run(service_instance, customer_id, run_id)
        resolved = requested_run or resolve_run_id(service_instance, None, customer_id=resolved_customer)

        results: list[dict[str, Any]] = []
        future = _RECENT_RESULTS_EXECUTOR.submit(
            service_instance.list_recent_results,
            limit,
            run_id=resolved,
            customer_id=resolved_customer,
            site_id=site_id,
        )
        try:
            fetched = future.result(timeout=_RECENT_RESULTS_TIMEOUT_S)
            results = fetched if isinstance(fetched, list) else []
        except FutureTimeoutError:
            future.cancel()
            logger.warning(
                "recent results retrieval timed out in %.1fms run_id=%s",
                _RECENT_RESULTS_TIMEOUT_S * 1000.0,
                resolved,
            )
            results = []
        except Exception:
            logger.exception("recent results retrieval failed run_id=%s", resolved)
            results = []

        if compact:
            compact_results = [_compact_result_view(r) for r in results]
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            logger.info("EXIT /results/recent compact=true duration_ms=%s", duration_ms)
            return JSONResponse(content={"count": len(compact_results), "results": compact_results})
        latest = results[0] if results else None

        duration_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info("EXIT /results/recent duration_ms=%s", duration_ms)
        return _results_envelope(results, latest=latest)

    @app.get("/results/export", response_model=ExportEnvelope)
    def export_results(
        format: Literal["json", "csv"] = Query(default="json"),
        limit: int = Query(default=500, ge=1, le=5000),
        run_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        site_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id_for_run(service_instance, customer_id, run_id)
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
