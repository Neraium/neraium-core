from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from .._core_imports import get_core_runtime_status

logger = logging.getLogger(__name__)
CMAPSS_DEFAULT_MAX_FRAMES = 240


def build_demo_router(*, service_instance: Any, require_api_key: Any, resolve_customer_id: Any, resolve_run_id_with_default: Any, run_demo_seed_job: Any, public_demo_job: Any, demo_jobs: dict[str, Any], demo_jobs_lock: Any, load_cmapss_fd004_subset: Any, log_structured: Any, summarize_exception_for_logs: Any, models: Any, utc_now_iso: Any) -> APIRouter:
    router = APIRouter(tags=["demo"])

    @router.post("/demo/seed/start")
    def demo_seed_start(payload: models.DemoSeedRequest, _: None = Depends(require_api_key), run_id: str | None = Query(default=None), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id or payload.customer_id)
        resolved_run = resolve_run_id_with_default(service_instance, run_id or payload.run_id, customer_id=resolved_customer)
        job_id = str(uuid4())
        now = utc_now_iso()
        with demo_jobs_lock:
            demo_jobs[job_id] = {"job_id": job_id, "status": "pending", "run_id": resolved_run, "customer_id": resolved_customer, "progress": 0, "processed": 0, "total_frames": int(payload.minutes), "message": "Preparing demo run...", "error": None, "created_at": now, "updated_at": now}
        threading.Thread(target=run_demo_seed_job, kwargs={"job_id": job_id, "resolved_run": resolved_run, "resolved_customer": resolved_customer, "payload": payload}, daemon=True).start()
        return {"status": "started", "job_id": job_id, "run_id": resolved_run, "message": "Demo seeding started."}

    @router.get("/demo/seed/status")
    def demo_seed_status(job_id: str = Query(..., min_length=1), _: None = Depends(require_api_key)) -> dict[str, Any]:
        with demo_jobs_lock:
            job = demo_jobs.get(job_id)
        if job is None:
            return {"status": "error", "job_id": job_id, "progress": 0, "run_id": "", "processed": 0, "total_frames": 0, "message": "Demo seed job not found.", "error": "job_not_found"}
        return public_demo_job(job)

    @router.post("/demo/cmapss/start")
    def demo_cmapss_start(payload: models.DemoCmapssStartRequest | None = None, _: None = Depends(require_api_key), customer_id: str | None = Query(default=None)) -> dict[str, Any]:
        request_payload = payload or models.DemoCmapssStartRequest()
        resolved_customer = resolve_customer_id(customer_id or request_payload.customer_id)
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
        log_structured(
            logger,
            event="demo_cmapss_start_requested",
            fields={
                "customer_id": resolved_customer,
                "requested_frames": requested_frames,
                "runtime_mode": runtime_status.get("mode", "unknown"),
            },
        )
        run = service_instance.create_run(
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
            rows = load_cmapss_fd004_subset(requested_frames)
            payload_rows = [{**row, "customer_id": resolved_customer} for row in rows]
            results = service_instance.ingest_batch(payload_rows, run_id=run_id, customer_id=resolved_customer)
        except Exception as exc:
            detail = summarize_exception_for_logs(exc)
            log_structured(logger, event="demo_cmapss_start_failure", fields={"run_id": run_id, "customer_id": resolved_customer, "error": detail}, level=logging.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to run NASA CMAPSS FD004 demo: {detail}") from exc
        processed = len(results)
        log_structured(
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
        }

    @router.get("/demo/cmapss/status")
    def demo_cmapss_status(
        run_id: str = Query(..., min_length=1),
        customer_id: str | None = Query(default=None),
        _: None = Depends(require_api_key),
    ) -> dict[str, Any]:
        resolved_customer = resolve_customer_id(customer_id)
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

        run = service_instance.get_run(run_id, customer_id=resolved_customer)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Unknown run_id: {run_id}")
        requested_frames = int((run.get("config") or {}).get("requested_frames") or CMAPSS_DEFAULT_MAX_FRAMES)
        try:
            recent_results = service_instance.list_recent_results(
                limit=5,
                run_id=run_id,
                customer_id=resolved_customer,
            )
        except Exception as exc:
            detail = summarize_exception_for_logs(exc)
            log_structured(
                logger,
                event="demo_cmapss_status_results_fetch_failed",
                fields={"run_id": run_id, "customer_id": resolved_customer, "error": detail},
                level=logging.ERROR,
            )
            recent_results = []
        try:
            history = service_instance.get_recent_history(
                limit=5,
                run_id=run_id,
                customer_id=resolved_customer,
            )
        except Exception:
            history = []
        try:
            current_state = service_instance.get_current_state(
                run_id=run_id,
                customer_id=resolved_customer,
            )
        except Exception:
            current_state = None
        results_count = len(recent_results or [])
        history_count = len(history or [])
        has_state = isinstance(current_state, dict) and bool(current_state)
        frames_processed = int(results_count)
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

        log_structured(
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
        }

    return router
