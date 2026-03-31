from __future__ import annotations

import logging
import threading
from typing import Any

from .operational_state import OperationalStateService
from .state_store import RuntimeStateStore


class OperationalStateRestoreService:
    def __init__(
        self,
        *,
        operational_state: OperationalStateService,
        store: RuntimeStateStore,
        resolve_customer_id: Any,
        utc_now_iso: Any,
        pull_manager: Any,
        runtime_state_diagnostics: dict[str, Any],
        logger: logging.Logger,
        log_structured: Any,
    ) -> None:
        self._operational_state = operational_state
        self._store = store
        self._resolve_customer_id = resolve_customer_id
        self._utc_now_iso = utc_now_iso
        self._pull_manager = pull_manager
        self._runtime_state_diagnostics = runtime_state_diagnostics
        self._logger = logger
        self._log_structured = log_structured

    def restore(self) -> None:
        if not self._operational_state.enabled:
            return
        restored_alert_customers = 0
        restored_ingest_jobs = 0
        restored_pull_integrations = 0
        try:
            for row in self._operational_state.list(key_prefix="alerts:"):
                state = row.get("state")
                if not isinstance(state, dict):
                    continue
                customer = self._resolve_customer_id(state.get("customer_id") or row.get("customer_id"))
                items = state.get("alerts")
                if not isinstance(items, list):
                    continue
                with self._store.alerts_lock:
                    self._store.alerts[customer] = [dict(x) for x in items if isinstance(x, dict)][:200]
                restored_alert_customers += 1
        except Exception:
            self._logger.exception("Failed restoring persisted alerts state")
        try:
            for row in self._operational_state.list(key_prefix="ingest_job:"):
                state = row.get("state")
                if not isinstance(state, dict):
                    continue
                job_id = str(state.get("job_id") or "")
                if not job_id:
                    continue
                status = str(state.get("status") or "unknown")
                if status in {"uploading", "queued", "processing"}:
                    state["status"] = "failed"
                    state["message"] = "Job interrupted by process restart before completion."
                    state["updated_at"] = self._utc_now_iso()
                    self._operational_state.persist(
                        f"ingest_job:{job_id}",
                        state,
                        customer_id=self._resolve_customer_id(state.get("customer_id")),
                        run_id=str(state.get("run_id")) if state.get("run_id") is not None else None,
                    )
                with self._store.ingest_jobs_lock:
                    self._store.ingest_jobs[job_id] = dict(state)
                restored_ingest_jobs += 1
        except Exception:
            self._logger.exception("Failed restoring persisted ingest job state")
        try:
            for row in self._operational_state.list(key_prefix="pull_integration:"):
                state = row.get("state")
                if not isinstance(state, dict):
                    continue
                customer = self._resolve_customer_id(state.get("customer_id") or row.get("customer_id"))
                merged = self._pull_manager.default_pull_state(customer)
                merged.update({k: v for k, v in state.items() if not str(k).startswith("_")})
                should_resume = bool(state.get("resume_on_startup"))
                merged["running"] = should_resume
                merged["status"] = "running" if should_resume else str(merged.get("status") or "stopped")
                merged["message"] = (
                    "Pull integration resumed after restart."
                    if should_resume
                    else str(merged.get("message") or "Pull integration is stopped.")
                )
                merged["_stop_event"] = threading.Event() if should_resume else None
                merged["_thread"] = None
                merged["updated_at"] = self._utc_now_iso()
                with self._store.pull_integrations_lock:
                    self._store.pull_integrations[customer] = merged
                self._pull_manager.persist_pull_state(customer, merged)
                if should_resume:
                    self._pull_manager.start_pull_worker(customer)
                restored_pull_integrations += 1
        except Exception:
            self._logger.exception("Failed restoring persisted pull integration state")
        self._runtime_state_diagnostics["restored_state_counts"] = {
            "alert_customers": restored_alert_customers,
            "ingest_jobs": restored_ingest_jobs,
            "pull_integrations": restored_pull_integrations,
        }
        self._log_structured(
            self._logger,
            event="operational_state_restore_complete",
            fields=self._runtime_state_diagnostics.get("restored_state_counts") or {},
            level=logging.INFO,
        )
