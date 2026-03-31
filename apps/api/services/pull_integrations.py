from __future__ import annotations

import base64
import json
import logging
import threading
import urllib.request
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException

from .operational_state import OperationalStateService
from .state_store import RuntimeStateStore


class PullIntegrationsManager:
    def __init__(
        self,
        *,
        app: Any,
        store: RuntimeStateStore,
        service_instance: Any,
        operational_state: OperationalStateService,
        resolve_customer_id: Any,
        utc_now_iso: Any,
        safe_float: Any,
        parse_int: Any,
        parse_finite_float: Any,
        log_structured: Any,
        summarize_exception_for_logs: Any,
        load_integration_config: Any,
        apply_integration_mapping: Any,
        integration_mapping_error: type[Exception],
        logger: logging.Logger,
    ) -> None:
        self.app = app
        self.store = store
        self._service_instance = service_instance
        self._operational_state = operational_state
        self._resolve_customer_id = resolve_customer_id
        self._utc_now_iso = utc_now_iso
        self._safe_float = safe_float
        self.parse_int = parse_int
        self.parse_finite_float = parse_finite_float
        self._log_structured = log_structured
        self._summarize_exception_for_logs = summarize_exception_for_logs
        self._load_integration_config = load_integration_config
        self._apply_integration_mapping = apply_integration_mapping
        self._integration_mapping_error = integration_mapping_error
        self._logger = logger

    def default_pull_state(self, customer_id: str) -> dict[str, Any]:
        now = self._utc_now_iso()
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

    def public_pull_state(self, state: dict[str, Any] | None, *, customer_id: str) -> dict[str, Any]:
        base = self.default_pull_state(customer_id)
        private_keys = {"username", "password", "token"}
        if state is None:
            return {k: v for k, v in base.items() if not k.startswith("_") and k not in private_keys}
        merged = dict(base)
        merged.update(state)
        return {k: v for k, v in merged.items() if not k.startswith("_") and k not in private_keys}

    def persist_pull_state(self, customer_id: str, state: dict[str, Any]) -> None:
        self._operational_state.persist_pull_state(
            customer_id,
            state,
            public_state_builder=self.public_pull_state,
        )

    @staticmethod
    def validate_endpoint_url(endpoint_url: str) -> str:
        text = str(endpoint_url or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="endpoint_url is required.")
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="endpoint_url must be a valid http(s) URL.")
        return text

    def _pull_auth_header(self, state: dict[str, Any]) -> str | None:
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

    def _coerce_pull_items(self, payload: Any, *, customer_id: str) -> list[dict[str, Any]]:
        cfg_override = getattr(self.app.state, "integration_config_override", None)
        if isinstance(cfg_override, dict):
            cfg = cfg_override
        else:
            path_override = getattr(self.app.state, "integration_config_path_override", None)
            cfg = self._load_integration_config(str(path_override or "").strip() or None)
        try:
            return self._apply_integration_mapping(payload, customer_id=customer_id, config=cfg)
        except self._integration_mapping_error as exc:
            raise ValueError(str(exc)) from exc

    def _fetch_pull_payload(self, state: dict[str, Any]) -> tuple[int, Any]:
        endpoint_url = str(state.get("endpoint_url") or "").strip()
        timeout_s = float(state.get("request_timeout_seconds") or 10.0)
        headers = {"Accept": "application/json"}
        auth_header = self._pull_auth_header(state)
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

    def _ingest_pull_items(self, *, rows: list[dict[str, Any]], run_id: str, customer_id: str) -> int:
        if not rows:
            return 0
        from neraium_core.ingestion_normalization import normalize_external_batch_payload

        normalized, _ = normalize_external_batch_payload(rows, customer_id=customer_id)
        results = self._service_instance.ingest_normalized_frames(normalized, run_id=run_id, customer_id=customer_id)
        return len(results)


    def start_pull_integration(
        self,
        *,
        customer_id: str,
        endpoint_url: str,
        run_id: str,
        auth_type: str,
        username: str | None,
        password: str | None,
        token: str | None,
        polling_interval_seconds: float,
        retry_max_attempts: int,
        retry_backoff_seconds: float,
        request_timeout_seconds: float,
    ) -> dict[str, Any]:
        resolved_customer = self._resolve_customer_id(customer_id)
        self.stop_pull_integration(resolved_customer, reason="Restarting integration.")
        stop_event = threading.Event()
        started_at = self._utc_now_iso()
        with self.store.pull_integrations_lock:
            self.store.pull_integrations[resolved_customer] = {
                "customer_id": resolved_customer,
                "endpoint_url": endpoint_url,
                "run_id": run_id,
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
            state = dict(self.store.pull_integrations[resolved_customer])
        self.persist_pull_state(resolved_customer, state)
        self.start_pull_worker(resolved_customer)
        return state

    def stop_pull_integration(self, customer_id: str, *, reason: str) -> dict[str, Any]:
        resolved_customer = self._resolve_customer_id(customer_id)
        stop_event: threading.Event | None = None
        thread: threading.Thread | None = None
        with self.store.pull_integrations_lock:
            state = self.store.pull_integrations.get(resolved_customer)
            if state is None:
                return self.public_pull_state(None, customer_id=resolved_customer)
            stop_event = state.get("_stop_event")
            thread = state.get("_thread")
            state["running"] = False
            state["status"] = "stopped"
            state["message"] = reason
            state["updated_at"] = self._utc_now_iso()
            self.persist_pull_state(resolved_customer, state)
        if isinstance(stop_event, threading.Event):
            stop_event.set()
        if isinstance(thread, threading.Thread) and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self.store.pull_integrations_lock:
            final_state = self.store.pull_integrations.get(resolved_customer)
            return self.public_pull_state(final_state, customer_id=resolved_customer)

    def start_pull_worker(self, customer_id: str) -> None:
        resolved_customer = self._resolve_customer_id(customer_id)

        def _worker() -> None:
            while True:
                with self.store.pull_integrations_lock:
                    state = self.store.pull_integrations.get(resolved_customer)
                    if state is None:
                        return
                    stop_event = state.get("_stop_event")
                    is_running = bool(state.get("running"))
                    endpoint_url = str(state.get("endpoint_url") or "")
                    poll_interval = self._safe_float(state.get("polling_interval_seconds"), 30.0)
                    retry_attempts = max(1, self.parse_int(state.get("retry_max_attempts") or 3, field_name="retry_max_attempts"))
                    retry_backoff = self._safe_float(state.get("retry_backoff_seconds"), 1.0)
                    run_id = str(state.get("run_id") or "")
                    if is_running:
                        state["status"] = "running"
                        state["updated_at"] = self._utc_now_iso()
                        state["message"] = "Polling upstream API."
                        self.persist_pull_state(resolved_customer, state)
                if not is_running or not isinstance(stop_event, threading.Event) or stop_event.is_set():
                    return
                if not endpoint_url or not run_id:
                    with self.store.pull_integrations_lock:
                        current = self.store.pull_integrations.get(resolved_customer)
                        if current is not None:
                            current["running"] = False
                            current["status"] = "error"
                            current["message"] = "Integration misconfigured: missing endpoint or run_id."
                            current["last_error"] = current["message"]
                            current["updated_at"] = self._utc_now_iso()
                            self.persist_pull_state(resolved_customer, current)
                    return
                success = False
                for attempt in range(1, max(1, retry_attempts) + 1):
                    with self.store.pull_integrations_lock:
                        current = self.store.pull_integrations.get(resolved_customer)
                        if current is None:
                            return
                        current["last_poll_at"] = self._utc_now_iso()
                        current["total_polls"] = int(current.get("total_polls", 0)) + 1
                        current["updated_at"] = self._utc_now_iso()
                        self.persist_pull_state(resolved_customer, current)
                    try:
                        http_status, payload = self._fetch_pull_payload(state)
                        rows = self._coerce_pull_items(payload, customer_id=resolved_customer)
                        ingested = self._ingest_pull_items(rows=rows, run_id=run_id, customer_id=resolved_customer)
                        now = self._utc_now_iso()
                        with self.store.pull_integrations_lock:
                            current = self.store.pull_integrations.get(resolved_customer)
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
                            self.persist_pull_state(resolved_customer, current)
                        self._log_structured(
                            self._logger,
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
                        with self.store.pull_integrations_lock:
                            current = self.store.pull_integrations.get(resolved_customer)
                            if current is None:
                                return
                            current["total_failures"] = int(current.get("total_failures", 0)) + 1
                            current["consecutive_failures"] = int(current.get("consecutive_failures", 0)) + 1
                            current["status"] = "error"
                            current["last_error"] = last_error
                            current["message"] = f"Poll attempt {attempt}/{retry_attempts} failed: {last_error}"
                            current["updated_at"] = self._utc_now_iso()
                            self.persist_pull_state(resolved_customer, current)
                        self._log_structured(
                            self._logger,
                            event="pull_integration_poll_failure",
                            fields={
                                "customer_id": resolved_customer,
                                "run_id": run_id,
                                "attempt": attempt,
                                "retry_attempts": retry_attempts,
                                "error": last_error,
                                **self._summarize_exception_for_logs(exc),
                            },
                            level=logging.WARNING,
                        )
                        if attempt >= retry_attempts:
                            break
                        delay = max(0.05, self._safe_float(retry_backoff, 1.0)) * (2 ** (attempt - 1))
                        if stop_event.wait(delay):
                            return
                if stop_event.wait(max(0.2, self._safe_float(poll_interval, 30.0))):
                    return
                if not success:
                    with self.store.pull_integrations_lock:
                        current = self.store.pull_integrations.get(resolved_customer)
                        if current is not None:
                            current["message"] = "Polling will continue after previous failure."
                            current["updated_at"] = self._utc_now_iso()
                            self.persist_pull_state(resolved_customer, current)

        worker = threading.Thread(target=_worker, daemon=True, name=f"pull-integration-{resolved_customer}")
        with self.store.pull_integrations_lock:
            state = self.store.pull_integrations.get(resolved_customer)
            if state is None:
                return
            state["_thread"] = worker
        worker.start()
