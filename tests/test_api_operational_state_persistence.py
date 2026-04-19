from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_client(db_path: str) -> TestClient:
    store = ResultStore(db_path=db_path)
    service = StructuralMonitoringService(engine=StructuralEngine(baseline_window=5, recent_window=3), store=store)
    return TestClient(create_app(service=service))


def _customer_path(path: str, customer_id: str = "persist-customer") -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}customer_id={customer_id}"


class _PullServer:
    def __init__(self, body_text: str):
        self.body_text = body_text
        self.requests = 0
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parent.requests += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(parent.body_text.encode("utf-8"))

            def log_message(self, format, *args):  # noqa: A003
                return

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _, port = s.getsockname()
        self._httpd = HTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{port}/pull"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)


def _create_run(client: TestClient, *, customer_id: str) -> str:
    response = client.post(
        _customer_path("/runs", customer_id=customer_id),
        json={"name": "persist-run", "activate": True, "config": {}},
    )
    assert response.status_code == 200
    return str(response.json()["run"]["run_id"])


def test_operational_state_persists_across_app_restart(tmp_path) -> None:
    db_path = str(tmp_path / "operational_state.db")
    customer_id = "persist-customer"
    pull_server = _PullServer(
        '[{"timestamp":"2026-01-01T00:00:00+00:00","site_id":"x","asset_id":"y","sensor_values":{"s1":1.0}}]'
    )
    endpoint = pull_server.start()
    try:
        with _build_client(db_path) as client:
            run_id = _create_run(client, customer_id=customer_id)
            alert_created = client.post(_customer_path(f"/alerts/test?run_id={run_id}", customer_id=customer_id))
            assert alert_created.status_code == 200

            started = client.post(
                _customer_path("/integrations/pull/start", customer_id=customer_id),
                json={
                    "endpoint_url": endpoint,
                    "run_id": run_id,
                    "polling_interval_seconds": 0.2,
                    "retry_max_attempts": 1,
                    "retry_backoff_seconds": 0.05,
                    "request_timeout_seconds": 2.0,
                },
            )
            assert started.status_code == 200
            time.sleep(0.35)

        with _build_client(db_path) as restarted:
            alerts = restarted.get(_customer_path(f"/alerts?run_id={run_id}&limit=20", customer_id=customer_id))
            assert alerts.status_code == 200
            items = alerts.json()["alerts"]
            assert any(str(item.get("type")) == "test_alert" for item in items)

            pull_state = restarted.get(_customer_path("/integrations/pull/status", customer_id=customer_id))
            assert pull_state.status_code == 200
            assert bool(pull_state.json().get("running")) is True

            health = restarted.get("/health")
            assert health.status_code == 200
            diagnostics = health.json().get("runtime_state_diagnostics") or {}
            assert diagnostics.get("persisted_state_enabled") is True
            assert "demo_jobs" in list(diagnostics.get("memory_only_state") or [])
    finally:
        pull_server.stop()
