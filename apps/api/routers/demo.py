from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

logger = logging.getLogger(__name__)


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
        run = service_instance.create_run(name=f"NASA CMAPSS FD004 Demo {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", config={"source": "nasa-cmapss-fd004", "dataset": "NASA CMAPSS FD004", "demo": "cmapss_fd004", "historical_run_replay": True}, activate=True, customer_id=resolved_customer)
        run_id = str(run.get("run_id") or "")
        if not run_id:
            raise HTTPException(status_code=500, detail="Failed to create demo run.")
        try:
            rows = load_cmapss_fd004_subset(request_payload.max_frames)
            payload_rows = [{**row, "customer_id": resolved_customer} for row in rows]
            results = service_instance.ingest_batch(payload_rows, run_id=run_id, customer_id=resolved_customer)
        except Exception as exc:
            detail = summarize_exception_for_logs(exc)
            log_structured(logger, event="demo_cmapss_start_failure", fields={"run_id": run_id, "customer_id": resolved_customer, "error": detail}, level=logging.ERROR)
            raise HTTPException(status_code=500, detail=f"Failed to run NASA CMAPSS FD004 demo: {detail}") from exc
        return {"status": "ok", "run_id": run_id, "processed": len(results), "demo": "cmapss_fd004"}

    return router
