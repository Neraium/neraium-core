from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from .._core_imports import get_core_runtime_status
from ..schemas.ingest import DemoCmapssStartRequest, DemoSeedRequest
from .dependencies import DemoRouterDependencies

_GROW_OP_SCENARIO_PATH = Path(__file__).resolve().parent.parent.parent.parent / "examples" / "demo" / "cannabis_grow_op_scenario.json"


def _load_grow_op_frames(customer_id: str) -> tuple[str, str, list[dict[str, Any]]]:
    """Load and flatten grow op scenario frames into ingestable payloads."""
    raw = json.loads(_GROW_OP_SCENARIO_PATH.read_text(encoding="utf-8"))
    asset = raw.get("asset") or {}
    site_id = str(asset.get("site_id") or "grow-op-facility-01")
    asset_id = str(asset.get("asset_id") or "canopy-zone-A")
    base_time = datetime.now(timezone.utc) - timedelta(minutes=1950)  # ~32 hrs back
    rows: list[dict[str, Any]] = []
    for phase in raw.get("phases") or []:
        for frame in phase.get("frames") or []:
            sensor_values = frame.get("sensor_values")
            if not isinstance(sensor_values, dict):
                continue
            ts = (base_time + timedelta(minutes=int(frame.get("minute_offset", 0)))).isoformat()
            rows.append({
                "timestamp": ts,
                "site_id": site_id,
                "asset_id": asset_id,
                "customer_id": customer_id,
                "sensor_values": {k: float(v) for k, v in sensor_values.items()},
            })
    return site_id, asset_id, rows

logger = logging.getLogger(__name__)

_GROW_OP_SCENARIO_PATH = Path(__file__).resolve().parent.parent / "demo_data" / "cannabis_grow_op_scenario.json"


def _load_grow_op_frames(customer_id: str) -> tuple[str, str, list[dict[str, Any]]]:
    """Load and flatten grow op scenario frames into ingestable payloads."""
    raw = json.loads(_GROW_OP_SCENARIO_PATH.read_text(encoding="utf-8"))
    asset = raw.get("asset") or {}
    site_id = str(asset.get("site_id") or "grow-op-facility-01")
    asset_id = str(asset.get("asset_id") or "canopy-zone-A")
    max_offset = max(
        (int(frame.get("minute_offset", 0)) for phase in raw.get("phases") or [] for frame in phase.get("frames") or []),
        default=6735,
    )
    base_time = datetime.now(timezone.utc) - timedelta(minutes=max_offset + 5)
    rows: list[dict[str, Any]] = []
    for phase in raw.get("phases") or []:
        for frame in phase.get("frames") or []:
            sensor_values = frame.get("sensor_values")
            if not isinstance(sensor_values, dict):
                continue
            ts = (base_time + timedelta(minutes=int(frame.get("minute_offset", 0)))).isoformat()
            rows.append({
                "timestamp": ts,
                "site_id": site_id,
                "asset_id": asset_id,
                "customer_id": customer_id,
                "sensor_values": {k: float(v) for k, v in sensor_values.items()},
            })
    return site_id, asset_id, rows
CMAPSS_DEFAULT_MAX_FRAMES = 240
CMAPSS_MIN_FRAMES = 30
CMAPSS_MAX_FRAMES = 500


def _canonical_demo_stage(*, frames_processed: int, risk_level: str, run_status: str) -> str:
    risk = str(risk_level or "UNKNOWN").upper()
    run = str(run_status or "").lower()
    if frames_processed <= 0 and run in {"starting", "pending", "queued", "initializing", "created", "open"}:
        return "warmup"
    if risk in {"LOW", "UNKNOWN"}:
        return "normal"
    if risk == "MEDIUM":
        return "early_structural_drift"
    if risk == "HIGH":
        return "rising_instability_actionable"
    return "stabilizing"


def _canonical_demo_message(*, stage: str, risk_level: str, trend: str) -> dict[str, Any]:
    normalized_risk = str(risk_level or "UNKNOWN").upper()
    normalized_trend = str(trend or "UNKNOWN").upper()
    if stage == "warmup":
        return {
            "what_is_happening": "Replay started and is building enough history for structural interpretation.",
            "why_we_believe_this": "The model needs baseline and recent windows before relational drift can be scored.",
            "operator_next_step": "Continue replay for at least one full warmup window.",
            "confidence": 0.55,
            "system_not_claiming": "No incident is being claimed during warmup.",
        }
    if stage == "normal":
        return {
            "what_is_happening": "System structure is currently stable.",
            "why_we_believe_this": "Composite instability and drift remain in low-risk bands.",
            "operator_next_step": "Continue monitoring and keep ingest uninterrupted.",
            "confidence": 0.72,
            "system_not_claiming": "This does not claim zero risk; only no current structural escalation.",
        }
    if stage == "early_structural_drift":
        return {
            "what_is_happening": "Early structural drift is present before hard threshold alarms.",
            "why_we_believe_this": f"Risk is {normalized_risk} with trend {normalized_trend}, indicating relationship-level change.",
            "operator_next_step": "Inspect implicated subsystem and increase observation cadence.",
            "confidence": 0.8,
            "system_not_claiming": "This is advisory and not an automated actuation decision.",
        }
    if stage == "rising_instability_actionable":
        return {
            "what_is_happening": "Instability is rising and operational risk is actionable.",
            "why_we_believe_this": f"Risk is {normalized_risk} with persistent structural deterioration signals.",
            "operator_next_step": "Execute site procedure for high-risk review and targeted inspection.",
            "confidence": 0.87,
            "system_not_claiming": "This is not a guaranteed failure-time prediction.",
        }
    return {
        "what_is_happening": "Structural conditions are changing; continue monitored review.",
        "why_we_believe_this": "Recent telemetry indicates non-static relational behavior.",
        "operator_next_step": "Review current run detail and corroborate with domain procedures.",
        "confidence": 0.68,
        "system_not_claiming": "Neraium is read-only and does not actuate infrastructure.",
    }


def build_demo_router(*, deps: DemoRouterDependencies) -> APIRouter:
    router = APIRouter(tags=["demo"])

    @router.post("/demo/seed/start")
    def demo_seed_start(payload: DemoSeedRequest, _: None = Depends(deps.require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id or payload.customer_id)
        resolved_run = deps.resolve_run_id_with_default(deps.service_instance, run_id or payload.run_id, customer_id=resolved_customer)
        job_id = str(uuid4())
        now = deps.utc_now_iso()
        with deps.demo_jobs_lock:
            deps.demo_jobs[job_id] = {"job_id": job_id, "status": "pending", "run_id": resolved_run, "customer_id": resolved_customer, "progress": 0, "processed": 0, "total_frames": int(payload.minutes), "message": "Preparing demo run...", "error": None, "created_at": now, "updated_at": now}
        deps.start_demo_seed_job(job_id=job_id, resolved_run=resolved_run, resolved_customer=resolved_customer, payload=payload)
        return {"status": "started", "job_id": job_id, "run_id": resolved_run, "message": "Demo seeding started."}

    @router.get("/demo/seed/status")
    def demo_seed_status(job_id: str = Query(..., min_length=1), _: None = Depends(deps.require_api_key)) -> dict[str, Any]:
        with deps.demo_jobs_lock:
            job = deps.demo_jobs.get(job_id)
        if job is None:
            return {"status": "error", "job_id": job_id, "progress": 0, "run_id": "", "processed": 0, "total_frames": 0, "message": "Demo seed job not found.", "error": "job_not_found"}
        return deps.public_demo_job(job)

    @router.post("/demo/cmapss/start")
    def demo_cmapss_start(payload: DemoCmapssStartRequest | None = None, _: None = Depends(deps.require_api_key), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        request_payload = payload or DemoCmapssStartRequest()
        resolved_customer = deps.resolve_customer_id(customer_id or request_payload.customer_id)
        runtime_status = get_core_runtime_status()
        if bool(runtime_status.get("using_fallback", False)):
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "NASA CMAPSS replay requires full core runtime modules; fallback mode is active.",
                    "type": "core_runtime_unavailable",
                    "stage": "demo_replay_start",
                    "actionable_detail": "Deploy with full neraium_core runtime (disable fallback mode) and retry replay.",
                    "runtime_notes": [str(x) for x in runtime_status.get("notes", [])],
                },
            )
        requested_frames = int(request_payload.max_frames or CMAPSS_DEFAULT_MAX_FRAMES)
        requested_frames = max(CMAPSS_MIN_FRAMES, min(CMAPSS_MAX_FRAMES, requested_frames))
        deps.log_structured(
            logger,
            event="demo_cmapss_start_requested",
            fields={
                "customer_id": resolved_customer,
                "requested_frames": requested_frames,
                "runtime_mode": runtime_status.get("mode", "unknown"),
            },
        )
        run = deps.service_instance.create_run(
            name=f"NASA CMAPSS FD004 Demo {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            config={
                "source": "nasa-cmapss-fd004",
                "dataset": "NASA CMAPSS FD004",
                "demo": "cmapss_fd004",
                "historical_run_replay": True,
                "requested_frames": requested_frames,
            },
            activate=True,
            customer_id=resolved_customer,
        )
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise HTTPException(status_code=500, detail="Failed to create demo run.")
        try:
            rows = deps.load_cmapss_fd004_subset(requested_frames)
            payload_rows = [{**row, "customer_id": resolved_customer} for row in rows]
            results = deps.service_instance.ingest_batch(payload_rows, run_id=run_id, customer_id=resolved_customer)
        except Exception as exc:
            detail = deps.summarize_exception_for_logs(exc)
            deps.log_structured(logger, event="demo_cmapss_start_failure", fields={"run_id": run_id, "customer_id": resolved_customer, "error": detail}, level=logging.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to run NASA CMAPSS FD004 demo: {detail}") from exc
        processed = len(results)
        deps.log_structured(
            logger,
            event="demo_cmapss_start_completed",
            fields={
                "run_id": run_id,
                "customer_id": resolved_customer,
                "requested_frames": requested_frames,
                "processed_frames": processed,
                "launch_succeeded": processed > 0,
            },
        )
        return {
            "status": "ok",
            "run_id": run_id,
            "processed": processed,
            "requested_frames": requested_frames,
            "launch_succeeded": processed > 0,
            "demo": "cmapss_fd004",
            "canonical_story": {
                "sequence": [
                    "normal",
                    "early_structural_drift",
                    "rising_instability_actionable",
                    "critical_review",
                ],
                "read_only": True,
                "non_actuating": True,
                "human_in_the_loop": True,
            },
        }

    @router.post("/demo/grow-op/start")
    def demo_grow_op_start(
        _: None = Depends(deps.require_api_key),
        customer_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id)
        try:
            site_id, asset_id, rows = _load_grow_op_frames(resolved_customer)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load grow op scenario: {exc}") from exc
        run = deps.service_instance.create_run(
            name=f"Grow Op Demo — 7-phase arc {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            config={"source": "grow-op-demo", "site_id": site_id, "asset_id": asset_id},
            activate=True,
            customer_id=resolved_customer,
        )
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise HTTPException(status_code=500, detail="Failed to create demo run.")
        job_id = str(uuid4())
        now = deps.utc_now_iso()
        with deps.demo_jobs_lock:
            deps.demo_jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "run_id": run_id,
                "customer_id": resolved_customer,
                "progress": 0,
                "processed": 0,
                "total_frames": len(rows),
                "message": "Preparing grow-op demo stream...",
                "error": None,
                "created_at": now,
                "updated_at": now,
            }

        warm_start_frames = min(36, len(rows))
        preloaded = 0
        if warm_start_frames > 0:
            try:
                seed_rows = rows[:warm_start_frames]
                seed_results = deps.service_instance.ingest_batch(seed_rows, run_id=run_id, customer_id=resolved_customer)
                preloaded = len(seed_results)
            except Exception as exc:
                with deps.demo_jobs_lock:
                    job = deps.demo_jobs.get(job_id)
                    if job is not None:
                        job.update(
                            status="error",
                            progress=0,
                            processed=0,
                            message="Grow-op demo warm start failed.",
                            error=str(exc),
                        )
                        job["updated_at"] = deps.utc_now_iso()
                raise HTTPException(status_code=500, detail="Grow-op demo warm start failed.") from exc
            with deps.demo_jobs_lock:
                job = deps.demo_jobs.get(job_id)
                if job is not None:
                    pct = int(round((preloaded / max(1, len(rows))) * 100))
                    job.update(
                        status="running" if preloaded < len(rows) else "complete",
                        progress=max(0, min(100, pct)),
                        processed=preloaded,
                        message="Grow-op demo telemetry warm start complete." if preloaded else "Streaming grow-op demo telemetry...",
                        error=None,
                    )
                    job["updated_at"] = deps.utc_now_iso()

        def _run_grow_op_seed_job() -> None:
            tail_rows = rows[preloaded:]
            if not tail_rows:
                with deps.demo_jobs_lock:
                    job = deps.demo_jobs.get(job_id)
                    if job is not None:
                        job.update(
                            status="complete",
                            progress=100,
                            processed=preloaded,
                            message="Grow-op demo stream complete.",
                            error=None,
                        )
                        job["updated_at"] = deps.utc_now_iso()
                return
            with deps.demo_jobs_lock:
                job = deps.demo_jobs.get(job_id)
                if job is not None:
                    job.update(status="running", message="Streaming grow-op demo telemetry...")
                    job["updated_at"] = deps.utc_now_iso()
            try:
                results = deps.service_instance.ingest_batch(tail_rows, run_id=run_id, customer_id=resolved_customer)
                with deps.demo_jobs_lock:
                    job = deps.demo_jobs.get(job_id)
                    if job is not None:
                        processed_total = preloaded + len(results)
                        job.update(
                            status="complete",
                            progress=100,
                            processed=processed_total,
                            message="Grow-op demo stream complete.",
                            error=None,
                        )
                        job["updated_at"] = deps.utc_now_iso()
            except Exception as exc:
                with deps.demo_jobs_lock:
                    job = deps.demo_jobs.get(job_id)
                    if job is not None:
                        job.update(
                            status="error",
                            message="Grow-op demo stream failed.",
                            error=str(exc),
                        )
                        job["updated_at"] = deps.utc_now_iso()

        threading.Thread(target=_run_grow_op_seed_job, daemon=True).start()

        return {
            "status": "started",
            "job_id": job_id,
            "run_id": run_id,
            "processed": preloaded,
            "demo": "grow-op",
            "message": "Grow-op demo seeding started.",
        }

    @router.get("/demo/grow-op/status")
    def demo_grow_op_status(
        run_id: str = Query(..., min_length=1),
        job_id: str | None = Query(default=None),
        customer_id: str | None = Query(default=None),
        _: None = Depends(deps.require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id)
        if job_id:
            with deps.demo_jobs_lock:
                job = deps.demo_jobs.get(job_id)
            if job and str(job.get("customer_id") or "") == resolved_customer and str(job.get("run_id") or "") == run_id:
                return {
                    "run_id": run_id,
                    "job": deps.public_demo_job(job),
                    "status": "ready" if str(job.get("status")) == "complete" else str(job.get("status") or "unknown"),
                }
        run = deps.service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        recent = deps.service_instance.list_recent_results(limit=5, run_id=run_id, customer_id=resolved_customer)
        state = deps.service_instance.get_current_state(run_id=run_id, customer_id=resolved_customer)
        return {
            "run_id": run_id,
            "status": "ready" if recent else "empty",
            "frames_processed": len(recent),
            "risk_level": str((state or {}).get("risk_level") or "UNKNOWN").upper(),
        }

    @router.get("/demo/cmapss/status")
    def demo_cmapss_status(
        run_id: str = Query(..., min_length=1),
        customer_id: str | None = Query(default=None),
        _: None = Depends(deps.require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id)
        runtime_status = get_core_runtime_status()
        if bool(runtime_status.get("using_fallback", False)):
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Analysis engine is unavailable while runtime fallback mode is active.",
                    "type": "core_runtime_unavailable",
                    "stage": "demo_replay_status",
                    "actionable_detail": "Restore full neraium_core runtime modules, then rerun the NASA CMAPSS replay.",
                    "runtime_notes": [str(x) for x in runtime_status.get("notes", [])],
                },
            )

        run = deps.service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        requested_frames = int((run.get("config") or {}).get("requested_frames") or CMAPSS_DEFAULT_MAX_FRAMES)
        try:
            recent_results = deps.service_instance.list_recent_results(
                limit=5,
                run_id=run_id,
                customer_id=resolved_customer,
            )
        except Exception as exc:
            detail = deps.summarize_exception_for_logs(exc)
            deps.log_structured(
                logger,
                event="demo_cmapss_status_results_fetch_failed",
                fields={"run_id": run_id, "customer_id": resolved_customer, "error": detail},
                level=logging.ERROR,
            )
            recent_results = []
        try:
            history = deps.service_instance.get_recent_history(
                limit=5,
                run_id=run_id,
                customer_id=resolved_customer,
            )
        except Exception:
            history = []
        try:
            current_state = deps.service_instance.get_current_state(
                run_id=run_id,
                customer_id=resolved_customer,
            )
        except Exception:
            current_state = None
        results_count = len(recent_results or [])
        history_count = len(history or [])
        has_state = isinstance(current_state, dict) and bool(current_state)
        frames_processed = int(results_count)
        risk_level = str((current_state or {}).get("risk_level") or "")
        trend = str((current_state or {}).get("trend") or "")
        launch_succeeded = frames_processed > 0
        if launch_succeeded:
            status = "ready"
            error_message = ""
        elif str(run.get("status", "")).lower() in {"failed", "error", "aborted", "cancelled"}:
            status = "failed"
            error_message = "Run entered a failed state."
        elif str(run.get("status", "")).lower() in {"completed", "complete", "done", "finished"}:
            status = "empty"
            error_message = "Replay completed but no analysis results were produced."
        elif str(run.get("status", "")).lower() in {"starting", "pending", "queued", "initializing", "created", "open"}:
            status = "starting"
            error_message = ""
        else:
            status = "ingesting"
            error_message = ""

        deps.log_structured(
            logger,
            event="demo_cmapss_status_checked",
            fields={
                "run_id": run_id,
                "customer_id": resolved_customer,
                "status": status,
                "run_status": str(run.get("status", "")),
                "frames_processed": frames_processed,
                "requested_frames": requested_frames,
                "has_state": has_state,
                "history_count": history_count,
                "results_count": results_count,
            },
        )

        stage = _canonical_demo_stage(
            frames_processed=frames_processed,
            risk_level=risk_level,
            run_status=str(run.get("status", "")),
        )
        canonical_message = _canonical_demo_message(stage=stage, risk_level=risk_level, trend=trend)
        return {
            "run_id": run_id,
            "launch_succeeded": launch_succeeded,
            "frames_requested": requested_frames,
            "frames_processed": frames_processed,
            "has_results": results_count > 0,
            "has_history": history_count > 0,
            "has_state": has_state,
            "status": status,
            "run_status": str(run.get("status", "")),
            "error_message": error_message,
            "canonical_story_stage": stage,
            "risk_level": (risk_level or "UNKNOWN").upper(),
            "trend": (trend or "UNKNOWN").upper(),
            "message": canonical_message,
        }

    @router.get("/demo/cmapss/proof-summary")
    def demo_cmapss_proof_summary(
        run_id: str = Query(..., min_length=1),
        customer_id: str | None = Query(default=None),
        _: None = Depends(deps.require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = deps.resolve_customer_id(customer_id)
        run = deps.service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        recent_results = deps.service_instance.list_recent_results(limit=64, run_id=run_id, customer_id=resolved_customer)
        if not recent_results:
            return {
                "run_id": run_id,
                "story": _canonical_demo_message(stage="warmup", risk_level="UNKNOWN", trend="UNKNOWN"),
                "proof": {
                    "first_medium_risk_index": None,
                    "first_high_risk_index": None,
                    "threshold_detection_index": None,
                    "neraium_lead_vs_threshold_frames": None,
                },
            }
        first_medium = next((i for i, row in enumerate(recent_results) if str(row.get("risk_level", "")).upper() == "MEDIUM"), None)
        first_high = next((i for i, row in enumerate(recent_results) if str(row.get("risk_level", "")).upper() == "HIGH"), None)
        threshold_idx = next(
            (
                i
                for i, row in enumerate(recent_results)
                if float((row.get("experimental_analytics") or {}).get("composite_instability") or 0.0) >= 0.85
            ),
            None,
        )
        lead = (
            int(threshold_idx) - int(first_medium)
            if first_medium is not None and threshold_idx is not None
            else None
        )
        latest = recent_results[-1]
        latest_stage = _canonical_demo_stage(
            frames_processed=len(recent_results),
            risk_level=str(latest.get("risk_level") or "UNKNOWN"),
            run_status=str(run.get("status") or ""),
        )
        return {
            "run_id": run_id,
            "story": _canonical_demo_message(
                stage=latest_stage,
                risk_level=str(latest.get("risk_level") or "UNKNOWN"),
                trend=str(latest.get("trend") or "UNKNOWN"),
            ),
            "proof": {
                "first_medium_risk_index": first_medium,
                "first_high_risk_index": first_high,
                "threshold_detection_index": threshold_idx,
                "neraium_lead_vs_threshold_frames": lead,
            },
        }

    return router
