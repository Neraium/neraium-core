from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from .operational_state import OperationalStateService
from .state_store import RuntimeStateStore

DEFAULT_UPLOAD_STREAM_CHUNK_BYTES = 1024 * 1024
DEFAULT_INGEST_JOB_MAX_ERROR_SAMPLES = 25


class IngestJobsManager:
    def __init__(
        self,
        *,
        store: RuntimeStateStore,
        service_instance: Any,
        operational_state: OperationalStateService,
        resolve_customer_id: Any,
        utc_now_iso: Any,
        process_alerts_after_ingest: Any,
        actionable_validation_detail: Any,
        log_structured: Any,
        summarize_exception_for_logs: Any,
        logger: logging.Logger,
    ) -> None:
        self.store = store
        self._service_instance = service_instance
        self._operational_state = operational_state
        self._resolve_customer_id = resolve_customer_id
        self._utc_now_iso = utc_now_iso
        self._process_alerts_after_ingest = process_alerts_after_ingest
        self._actionable_validation_detail = actionable_validation_detail
        self._log_structured = log_structured
        self._summarize_exception_for_logs = summarize_exception_for_logs
        self._logger = logger

    def public_ingest_job(self, job: dict[str, Any]) -> dict[str, Any]:
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
            "customer_id": self._resolve_customer_id(job.get("customer_id")),
            "filename": str(job.get("filename") or "upload.csv"),
            "created_at": str(job.get("created_at") or self._utc_now_iso()),
            "updated_at": str(job.get("updated_at") or self._utc_now_iso()),
            "rows_processed": int(job.get("rows_processed", 0)),
            "rows_succeeded": int(job.get("rows_succeeded", 0)),
            "rows_failed": int(job.get("rows_failed", 0)),
            "partial_success": bool(job.get("partial_success", False)),
            "upload_bytes_received": int(job.get("upload_bytes_received", 0)),
            "upload_bytes_total": int(job.get("upload_bytes_total")) if job.get("upload_bytes_total") is not None else None,
            "error_samples": list(job.get("error_samples") or []),
            "message": job.get("message"),
            "latest_result": job.get("latest_result"),
            "lifecycle_phase": job.get("lifecycle_phase"),
            "ui_state": ui_state,
            "terminal_state": job.get("terminal_state"),
            "failure_category": job.get("failure_category"),
        }

    def cleanup_ingest_jobs(self, max_jobs: int = 300) -> None:
        removed_ids: list[str] = []
        with self.store.ingest_jobs_lock:
            if len(self.store.ingest_jobs) <= max_jobs:
                return
            completed_ids = [
                jid
                for jid, job in self.store.ingest_jobs.items()
                if str(job.get("status")) in {"completed", "partial_success", "failed"}
            ]
            completed_ids.sort(
                key=lambda jid: str(
                    self.store.ingest_jobs[jid].get("updated_at")
                    or self.store.ingest_jobs[jid].get("created_at")
                    or ""
                )
            )
            overflow = len(self.store.ingest_jobs) - max_jobs
            for jid in completed_ids[: max(0, overflow)]:
                self.store.ingest_jobs.pop(jid, None)
                removed_ids.append(jid)
        for jid in removed_ids:
            self._operational_state.delete(f"ingest_job:{jid}")

    def update_ingest_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.store.ingest_jobs_lock:
            job = self.store.ingest_jobs.get(job_id)
            if job is None:
                return None
            current_status = str(job.get("status") or "")
            next_status = str(fields.get("status") or current_status)
            terminal_statuses = {"completed", "partial_success", "failed"}
            if current_status in terminal_statuses and next_status not in terminal_statuses:
                return dict(job)
            job.update(fields)
            job["updated_at"] = self._utc_now_iso()
            if "partial_success" not in fields:
                job["partial_success"] = int(job.get("rows_succeeded", 0)) > 0 and int(job.get("rows_failed", 0)) > 0
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
            self._operational_state.persist(
                f"ingest_job:{job_id}",
                dict(job),
                customer_id=self._resolve_customer_id(job.get("customer_id")),
                run_id=str(job.get("run_id")) if job.get("run_id") is not None else None,
            )
            return dict(job)

    async def stream_upload_to_tempfile(self, upload: Any, target_path: Path, job_id: str, *, max_bytes: int | None = None) -> int:
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
                    if max_bytes is not None and bytes_received > int(max_bytes):
                        raise ValueError(f"Request body too large (max {int(max_bytes)} bytes).")
                    self.update_ingest_job(
                        job_id,
                        status="uploading",
                        upload_bytes_received=bytes_received,
                        message=f"Uploading CSV ({bytes_received} bytes received)...",
                    )
                out.flush()
        finally:
            await upload.close()
        return bytes_received

    def start_ingest_job_worker(
        self,
        *,
        job_id: str,
        temp_path: str,
        run_id: str,
        customer_id: str,
        column_mapping: dict[str, Any] | None = None,
    ) -> None:
        def _worker() -> None:
            self.update_ingest_job(job_id, status="processing", message="Upload complete. Ingest processing started.")
            self._log_structured(
                self._logger,
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
                    self.update_ingest_job(
                        job_id,
                        status="processing",
                        rows_processed=int(progress.get("rows_processed", 0)),
                        rows_succeeded=int(progress.get("rows_succeeded", 0)),
                        rows_failed=int(progress.get("rows_failed", 0)),
                        error_samples=list(progress.get("error_samples") or []),
                        message=progress.get("message"),
                    )

                with Path(temp_path).open("r", encoding="utf-8", newline="") as stream:
                    summary = self._service_instance.ingest_csv_stream(
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
                self.update_ingest_job(
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
                    recent_for_alerts = self._service_instance.list_recent_results(
                        limit=2,
                        run_id=run_id,
                        customer_id=customer_id,
                    )
                    if len(recent_for_alerts) >= 2:
                        previous_for_alerts = recent_for_alerts[1]
                    self._process_alerts_after_ingest(
                        customer_id=customer_id,
                        run_id=run_id,
                        latest_result=latest_from_summary,
                        previous_result=previous_for_alerts,
                    )
                self._log_structured(
                    self._logger,
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
                message = self._actionable_validation_detail(str(exc))
                self.update_ingest_job(job_id, status="failed", message=message, failure_category="ingest_failed")
                self._log_structured(
                    self._logger,
                    event="ingest_job_failed",
                    fields={
                        "job_id": job_id,
                        "run_id": run_id,
                        "customer_id": customer_id,
                        "ingest_path": "csv_upload",
                        "stage": "failed",
                        "error_type": "ingest_failed",
                        **self._summarize_exception_for_logs(exc),
                    },
                    level=logging.ERROR,
                )
            finally:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    self._logger.debug("Unable to remove temp upload file: %s", temp_path, exc_info=True)
                self.cleanup_ingest_jobs()

        threading.Thread(target=_worker, daemon=True, name=f"ingest-job-{job_id}").start()
