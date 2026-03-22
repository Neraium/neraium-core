from __future__ import annotations

import base64
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import DEFAULT_MAX_REQUEST_BODY_BYTES, create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _client(tmp_path, *, max_request_body_bytes: int | None = None) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "test_mvp.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service, max_request_body_bytes=max_request_body_bytes)
    return TestClient(app)


def _generate_csv_rows(row_count: int) -> str:
    header = "timestamp,site_id,asset_id,s1\n"
    row = "2026-01-01T00:00:00+00:00,a,b,1\n"
    return header + (row * row_count)


def _run_and_ingest(client: TestClient, customer_id: str = "customer-a") -> tuple[str, int]:
    run = client.post(
        _customer_path("/runs", customer_id=customer_id),
        json={"name": "mvp-run", "activate": True, "config": {"baseline_window": 5, "recent_window": 3}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]
    ing = client.post(
        _customer_path(f"/ingest?run_id={run_id}", customer_id=customer_id),
        json={
            "timestamp": "2026-01-01T00:00:00+00:00",
            "customer_id": customer_id,
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"pressure": 51.2, "flow": 12.4},
        },
    )
    assert ing.status_code == 200
    rid = int(ing.json()["results"][0]["result_id"])
    return run_id, rid


def _customer_path(path: str, customer_id: str = "customer-a") -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}customer_id={customer_id}"


class _PullServer:
    def __init__(self, body_text: str, *, auth_header: str | None = None):
        self.body_text = body_text
        self.auth_header = auth_header
        self.requests = 0
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                parent.requests += 1
                if parent.auth_header:
                    provided = self.headers.get("Authorization")
                    if provided != parent.auth_header:
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"detail":"unauthorized"}')
                        return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(parent.body_text.encode("utf-8"))

            def log_message(self, format, *args):  # noqa: A003
                return

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            host, port = s.getsockname()
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


def _wait_for_pull_ingest(
    client: TestClient,
    *,
    customer_id: str,
    min_count: int,
    timeout_s: float = 4.0,
) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(_customer_path("/integrations/pull/status", customer_id=customer_id))
        assert resp.status_code == 200
        payload = resp.json()
        last = payload
        if int(payload.get("total_ingested", 0)) >= min_count:
            return payload
        time.sleep(0.08)
    raise AssertionError(f"pull integration did not ingest target rows; last={last}")


def _wait_for_ingest_job(
    client: TestClient,
    job_id: str,
    *,
    timeout_s: float = 8.0,
    customer_id: str = "customer-a",
) -> dict:
    deadline = time.monotonic() + timeout_s
    last: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(_customer_path(f"/ingest/jobs/{job_id}", customer_id=customer_id))
        assert resp.status_code == 200
        payload = resp.json()
        last = payload
        if payload.get("status") in {"completed", "partial_success", "failed"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"ingest job did not complete before timeout; last={last}")


def test_mvp_routes_available(tmp_path) -> None:
    client = _client(tmp_path)
    home = client.get("/")
    assert home.status_code == 200
    assert "text/html" in home.headers.get("content-type", "")
    js = client.get("/web/app.js")
    css = client.get("/web/styles.css")
    assert js.status_code == 200
    assert css.status_code == 200


def test_web_js_smoke_wiring_for_demo_critical_controls(tmp_path) -> None:
    client = _client(tmp_path)
    js = client.get("/web/app.js")
    assert js.status_code == 200
    source = js.text
    expected_tokens = [
        "#loadingOverlay",
        "#loadingMessage",
        "createToast(",
        "riskBadgeHtml(",
        "phaseBadgeHtml(",
        "#seedDemoBtn",
        "#runsSearchInput",
        "#runResultsSearchInput",
        "#runRangeControls [data-range]",
        "#uploadDropZone",
        "#selectedFileName",
        "#dashboardEmpty",
        "#runDetailEmpty",
        "#runResultsEmpty",
    ]
    for token in expected_tokens:
        assert token in source


def test_run_scoped_result_detail_and_recent(tmp_path) -> None:
    client = _client(tmp_path)
    run_id, result_id = _run_and_ingest(client)

    detail = client.get(_customer_path(f"/results/{result_id}?run_id={run_id}"))
    assert detail.status_code == 200
    result = detail.json()["result"]
    assert result["run_id"] == run_id
    assert result["result_id"] == result_id

    recent = client.get(_customer_path(f"/results/recent?run_id={run_id}&limit=10"))
    assert recent.status_code == 200
    body = recent.json()
    assert body["count"] >= 1
    assert body["results"][0]["run_id"] == run_id


def test_geometry_endpoints_expose_engine_derived_structure(tmp_path) -> None:
    client = _client(tmp_path)
    run = client.post(
        _customer_path("/runs", customer_id="customer-a"),
        json={"name": "mvp-run-geometry", "activate": True, "config": {"baseline_window": 5, "recent_window": 3}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]
    for i in range(8):
        ing = client.post(
            _customer_path(f"/ingest?run_id={run_id}", customer_id="customer-a"),
            json={
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "customer_id": "customer-a",
                "site_id": "site-a",
                "asset_id": "asset-a",
                "sensor_values": {"pressure": 51.2 + i * 0.1, "flow": 12.4 + i * 0.2},
            },
        )
        assert ing.status_code == 200
    result_id = int(ing.json()["results"][0]["result_id"])

    run_geom = client.get(_customer_path(f"/runs/{run_id}/geometry"))
    assert run_geom.status_code == 200
    run_payload = run_geom.json()
    assert run_payload["run_id"] == run_id
    assert run_payload["available"] is True
    assert isinstance(run_payload["nodes"], list)
    assert len(run_payload["nodes"]) >= 2
    assert isinstance(run_payload["edges"], list)
    assert run_payload["projection"]["is_visualization_projection"] is True
    assert "engine_fields" in run_payload["provenance"]
    assert "views" in run_payload
    assert "current" in run_payload["views"]
    assert "baseline" in run_payload["views"]
    assert run_payload["views"]["current"]["available"] is True
    assert "summary" in run_payload
    assert "unstable_nodes_current" in run_payload["summary"]

    node = run_payload["nodes"][0]
    assert set(node.keys()) >= {"id", "label", "position", "magnitude", "stress", "state"}
    assert set(node["position"].keys()) == {"x", "y", "z"}
    assert set(node.keys()) >= {"position_current", "position_baseline", "is_unstable"}
    assert set(node["position_current"].keys()) == {"x", "y", "z"}
    assert set(node["position_baseline"].keys()) == {"x", "y", "z"}

    result_geom = client.get(_customer_path(f"/results/{result_id}/geometry?run_id={run_id}"))
    assert result_geom.status_code == 200
    result_payload = result_geom.json()
    assert result_payload["result_id"] == result_id
    assert result_payload["available"] is True
    assert result_payload["metrics"]["state"] in {"STABLE", "WATCH", "ALERT", "NOMINAL_STRUCTURE"}


def test_export_json_and_csv(tmp_path) -> None:
    client = _client(tmp_path)
    run_id, _ = _run_and_ingest(client)

    export_json = client.get(_customer_path(f"/results/export?run_id={run_id}&format=json&limit=50"))
    assert export_json.status_code == 200
    j = export_json.json()
    assert j["format"] == "json"
    decoded = json.loads(j["content"])
    assert isinstance(decoded, list)
    assert decoded

    export_csv = client.get(_customer_path(f"/results/export?run_id={run_id}&format=csv&limit=50"))
    assert export_csv.status_code == 200
    c = export_csv.json()
    assert c["format"] == "csv"
    assert "result_id,run_id,timestamp" in c["content"]


def test_update_and_activate_run_routes(tmp_path) -> None:
    client = _client(tmp_path)
    r1 = client.post(_customer_path("/runs"), json={"name": "run-1", "activate": True, "config": {"x": 1}})
    r2 = client.post(_customer_path("/runs"), json={"name": "run-2", "activate": False, "config": {"x": 2}})
    assert r1.status_code == 200
    assert r2.status_code == 200
    run2 = r2.json()["run"]["run_id"]

    upd = client.patch(
        _customer_path(f"/runs/{run2}"),
        json={"name": "run-2b", "status": "open", "config": {"x": 3}},
    )
    assert upd.status_code == 200
    assert upd.json()["run"]["name"] == "run-2b"

    act = client.post(_customer_path(f"/runs/{run2}/activate"))
    assert act.status_code == 200
    assert act.json()["run"]["is_active"] is True

    active = client.get(_customer_path("/runs/active"))
    assert active.status_code == 200
    assert active.json()["run"]["run_id"] == run2


def test_request_body_limit_allows_50mb_and_rejects_over_limit(tmp_path) -> None:
    assert DEFAULT_MAX_REQUEST_BODY_BYTES >= 50 * 1024 * 1024

    # Use a much smaller limit in test to keep runtime/memory bounded while
    # exercising the same middleware code path.
    client = _client(tmp_path, max_request_body_bytes=2048)
    run = client.post(
        _customer_path("/runs"),
        json={"name": "size-limit-run", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    allowed_csv = _generate_csv_rows(30)
    allowed_payload = {"csv_text": allowed_csv}
    allowed_resp = client.post(_customer_path(f"/ingest/csv?run_id={run_id}"), json=allowed_payload)
    assert allowed_resp.status_code == 200
    assert allowed_resp.json()["count"] > 0

    too_large_csv = _generate_csv_rows(120)
    too_large_payload = {"csv_text": too_large_csv}
    too_large_resp = client.post(_customer_path(f"/ingest/csv?run_id={run_id}"), json=too_large_payload)
    assert too_large_resp.status_code == 413
    assert "Request body too large" in too_large_resp.json()["detail"]


def test_request_body_limit_short_circuits_content_length(tmp_path) -> None:
    client = _client(tmp_path, max_request_body_bytes=2048)
    run = client.post(
        _customer_path("/runs"),
        json={"name": "size-limit-run-header", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    oversized_csv = _generate_csv_rows(40)
    response = client.post(
        _customer_path(f"/ingest/csv?run_id={run_id}"),
        json={"csv_text": oversized_csv},
        headers={"content-length": str(2048 + 10)},
    )
    assert response.status_code == 413
    assert "Request body too large" in response.json()["detail"]


def test_customer_isolation_for_results_and_runs(tmp_path) -> None:
    client = _client(tmp_path)

    run_a = client.post(_customer_path("/runs", customer_id="cust-a"), json={"name": "run-a", "activate": True, "config": {}})
    run_b = client.post(_customer_path("/runs", customer_id="cust-b"), json={"name": "run-b", "activate": True, "config": {}})
    assert run_a.status_code == 200
    assert run_b.status_code == 200
    run_a_id = run_a.json()["run"]["run_id"]
    run_b_id = run_b.json()["run"]["run_id"]

    ing_a = client.post(
        _customer_path(f"/ingest?run_id={run_a_id}", customer_id="cust-a"),
        json={
            "timestamp": "2026-01-01T00:00:00+00:00",
            "customer_id": "cust-a",
            "site_id": "site-1",
            "asset_id": "asset-1",
            "sensor_values": {"pressure": 50.0, "flow": 10.0},
        },
    )
    ing_b = client.post(
        _customer_path(f"/ingest?run_id={run_b_id}", customer_id="cust-b"),
        json={
            "timestamp": "2026-01-01T00:01:00+00:00",
            "customer_id": "cust-b",
            "site_id": "site-2",
            "asset_id": "asset-2",
            "sensor_values": {"pressure": 60.0, "flow": 11.0},
        },
    )
    assert ing_a.status_code == 200
    assert ing_b.status_code == 200

    recent_a = client.get(_customer_path("/results/recent?limit=10", customer_id="cust-a"))
    recent_b = client.get(_customer_path("/results/recent?limit=10", customer_id="cust-b"))
    assert recent_a.status_code == 200
    assert recent_b.status_code == 200
    assert recent_a.json()["count"] == 1
    assert recent_b.json()["count"] == 1
    assert recent_a.json()["results"][0]["customer_id"] == "cust-a"
    assert recent_b.json()["results"][0]["customer_id"] == "cust-b"

    runs_a = client.get(_customer_path("/runs?limit=10", customer_id="cust-a"))
    runs_b = client.get(_customer_path("/runs?limit=10", customer_id="cust-b"))
    assert runs_a.status_code == 200
    assert runs_b.status_code == 200
    assert all(item["customer_id"] == "cust-a" for item in runs_a.json()["runs"])
    assert all(item["customer_id"] == "cust-b" for item in runs_b.json()["runs"])


def test_streamed_csv_upload_job_completes_and_tracks_progress(tmp_path) -> None:
    client = _client(tmp_path)
    run = client.post(
        _customer_path("/runs"),
        json={"name": "stream-upload-run", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    csv_bytes = (
        "timestamp,site_id,asset_id,s1,s2\n"
        "2026-01-01T00:00:00+00:00,site-a,asset-a,1.0,2.0\n"
        "2026-01-01T00:00:01+00:00,site-a,asset-a,1.1,2.1\n"
    ).encode("utf-8")
    start = client.post(
        _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
        files={"file": ("telemetry.csv", csv_bytes, "text/csv")},
    )
    assert start.status_code == 200
    started = start.json()
    assert started["job_id"]
    assert started["status"] in {"uploading", "queued", "processing", "completed"}
    assert started["upload_bytes_total"] is None or started["upload_bytes_total"] >= len(csv_bytes)

    done = _wait_for_ingest_job(client, started["job_id"])
    assert done["status"] == "completed"
    assert done["rows_processed"] == 2
    assert done["rows_succeeded"] == 2
    assert done["rows_failed"] == 0

    recent = client.get(_customer_path(f"/results/recent?run_id={run_id}&limit=10"))
    assert recent.status_code == 200
    assert recent.json()["count"] >= 2


def test_streamed_csv_upload_reports_partial_success(tmp_path) -> None:
    client = _client(tmp_path)
    run = client.post(
        _customer_path("/runs"),
        json={"name": "stream-upload-partial", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    csv_bytes = (
        "timestamp,site_id,asset_id,s1\n"
        "2026-01-01T00:00:00+00:00,site-a,asset-a,1.0\n"
        "not-a-timestamp,site-a,asset-a,1.1\n"
        "2026-01-01T00:00:02+00:00,site-a,asset-a,1.2\n"
    ).encode("utf-8")
    start = client.post(
        _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
        files={"file": ("telemetry.csv", csv_bytes, "text/csv")},
    )
    assert start.status_code == 200
    job_id = start.json()["job_id"]

    done = _wait_for_ingest_job(client, job_id)
    assert done["status"] == "partial_success"
    assert done["rows_processed"] == 3
    assert done["rows_succeeded"] == 2
    assert done["rows_failed"] == 1
    assert done["partial_success"] is True
    assert done["error_samples"]
    assert "Row" in str(done["error_samples"][0].get("message", ""))


def test_stream_upload_rejects_non_csv_extension(tmp_path) -> None:
    client = _client(tmp_path)
    run = client.post(
        _customer_path("/runs"),
        json={"name": "stream-upload-invalid-ext", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]
    resp = client.post(
        _customer_path(f"/ingest/csv/upload?run_id={run_id}"),
        files={"file": ("telemetry.txt", b"timestamp,site_id,asset_id,s1\n", "text/plain")},
    )
    assert resp.status_code == 400
    assert ".csv file" in resp.json()["detail"]


def test_pull_integration_start_status_stop_and_ingest(tmp_path) -> None:
    client = _client(tmp_path)
    customer_id = "pull-customer-a"
    run = client.post(
        _customer_path("/runs", customer_id=customer_id),
        json={"name": "pull-run", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    payload = json.dumps(
        {
            "items": [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"pressure": 50.0, "flow": 11.0},
                },
                {
                    "timestamp": "2026-01-01T00:00:01+00:00",
                    "site_id": "site-a",
                    "asset_id": "asset-a",
                    "sensor_values": {"pressure": 50.1, "flow": 11.1},
                },
            ]
        }
    )
    server = _PullServer(payload, auth_header="Bearer secret-token")
    endpoint = server.start()
    try:
        start = client.post(
            _customer_path("/integrations/pull/start", customer_id=customer_id),
            json={
                "endpoint_url": endpoint,
                "polling_interval_seconds": 0.2,
                "auth_type": "bearer",
                "token": "secret-token",
                "run_id": run_id,
                "retry_max_attempts": 2,
                "retry_backoff_seconds": 0.05,
                "request_timeout_seconds": 2.0,
            },
        )
        assert start.status_code == 200
        started = start.json()
        assert started["running"] is True
        assert started["status"] in {"running", "error"}
        assert started["endpoint_url"] == endpoint
        assert started["run_id"] == run_id

        pulled = _wait_for_pull_ingest(client, customer_id=customer_id, min_count=2)
        assert pulled["running"] is True
        assert int(pulled.get("total_ingested", 0)) >= 2
        assert pulled["last_success_at"] is not None
        assert pulled["last_error"] in {None, ""}

        stop = client.post(_customer_path("/integrations/pull/stop", customer_id=customer_id))
        assert stop.status_code == 200
        stopped = stop.json()
        assert stopped["running"] is False
        assert stopped["status"] == "stopped"

        recent = client.get(_customer_path(f"/results/recent?run_id={run_id}&limit=10", customer_id=customer_id))
        assert recent.status_code == 200
        assert recent.json()["count"] >= 2
    finally:
        server.stop()


def test_pull_integration_reports_failures_with_retries(tmp_path) -> None:
    client = _client(tmp_path)
    customer_id = "pull-customer-b"
    run = client.post(
        _customer_path("/runs", customer_id=customer_id),
        json={"name": "pull-run-fail", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    # Unbound localhost port should fail quickly and trigger retry accounting.
    endpoint = "http://127.0.0.1:9/pull"
    start = client.post(
        _customer_path("/integrations/pull/start", customer_id=customer_id),
        json={
            "endpoint_url": endpoint,
            "polling_interval_seconds": 0.2,
            "auth_type": "none",
            "run_id": run_id,
            "retry_max_attempts": 2,
            "retry_backoff_seconds": 0.05,
            "request_timeout_seconds": 1.0,
        },
    )
    assert start.status_code == 200

    deadline = time.monotonic() + 4.0
    last = {}
    while time.monotonic() < deadline:
        status_resp = client.get(_customer_path("/integrations/pull/status", customer_id=customer_id))
        assert status_resp.status_code == 200
        last = status_resp.json()
        if int(last.get("total_failures", 0)) >= 1:
            break
        time.sleep(0.08)

    assert int(last.get("total_failures", 0)) >= 1
    assert int(last.get("consecutive_failures", 0)) >= 1
    assert last.get("last_error")
    stop = client.post(_customer_path("/integrations/pull/stop", customer_id=customer_id))
    assert stop.status_code == 200


def test_pull_integration_with_basic_auth_and_mapping_config(tmp_path) -> None:
    client = _client(tmp_path)
    customer_id = "pull-customer-c"
    run = client.post(
        _customer_path("/runs", customer_id=customer_id),
        json={"name": "pull-run-basic-auth", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    mapping_path = tmp_path / "integration.json"
    mapping_path.write_text(
        json.dumps(
            {
                "customers": {
                    customer_id: {
                        "mapping": {
                            "payload_path": "data",
                            "items_path": "events",
                            "field_aliases": {
                                "timestamp": ["time stamp"],
                                "site_id": ["site code"],
                                "asset_id": ["asset code"],
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client.app.state.integration_config_path_override = str(mapping_path)
    client.app.state.integration_config_override = None
    try:
        payload = json.dumps(
            {
                "data": {
                    "events": [
                        {
                            "time stamp": "2026-01-01T00:00:00+00:00",
                            "site code": "site-auth",
                            "asset code": "asset-auth",
                            "Pressure PSI": 55.1,
                        }
                    ]
                }
            }
        )
        auth_header = "Basic " + base64.b64encode(b"demo-user:demo-pass").decode("ascii")
        server = _PullServer(payload, auth_header=auth_header)
        endpoint = server.start()
        try:
            start = client.post(
                _customer_path("/integrations/pull/start", customer_id=customer_id),
                json={
                    "endpoint_url": endpoint,
                    "polling_interval_seconds": 0.2,
                    "auth_type": "basic",
                    "username": "demo-user",
                    "password": "demo-pass",
                    "run_id": run_id,
                    "retry_max_attempts": 2,
                    "retry_backoff_seconds": 0.05,
                    "request_timeout_seconds": 2.0,
                },
            )
            assert start.status_code == 200
            pulled = _wait_for_pull_ingest(client, customer_id=customer_id, min_count=1)
            assert pulled["running"] is True
            recent = client.get(
                _customer_path(
                    f"/results/recent?run_id={run_id}&limit=5",
                    customer_id=customer_id,
                )
            )
            assert recent.status_code == 200
            assert recent.json()["count"] >= 1
            sensors = recent.json()["results"][0].get("sensor_relationships") or []
            assert "pressure_psi" in {str(k).lower() for k in sensors}
            stop = client.post(_customer_path("/integrations/pull/stop", customer_id=customer_id))
            assert stop.status_code == 200
        finally:
            server.stop()
    finally:
        client.app.state.integration_config_path_override = None
        client.app.state.integration_config_override = None


def test_pull_integration_applies_customer_mapping_config(tmp_path) -> None:
    mapping_path = tmp_path / "integration_config.json"
    mapping_path.write_text(
        json.dumps(
            {
                "customers": {
                    "pull-customer-map": {
                        "mapping": {
                            "payload_path": "payload",
                            "items_path": "rows",
                            "field_aliases": {
                                "timestamp": ["time_stamp"],
                                "site_id": ["SITE ID"],
                                "asset_id": ["asset id"],
                                "sensor_values": ["SENSORS"],
                            },
                            "sensor_aliases": {
                                "temp c": "temperature",
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    store = ResultStore(db_path=str(tmp_path / "test_pull_map.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service)
    app.state.integration_config_path_override = str(mapping_path)
    app.state.integration_config_override = None
    client = TestClient(app)

    customer_id = "pull-customer-map"
    run = client.post(
        _customer_path("/runs", customer_id=customer_id),
        json={"name": "pull-map-run", "activate": True, "config": {}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]

    payload = json.dumps(
        {
            "payload": {
                "rows": [
                    {
                        "time_stamp": "2026-01-01T00:00:00+00:00",
                        "SITE ID": "site-map",
                        "asset id": "asset-map",
                        "SENSORS": {"temp c": 61.2, "pressure": 42.1},
                    }
                ]
            }
        }
    )
    server = _PullServer(payload)
    endpoint = server.start()
    try:
        start = client.post(
            _customer_path("/integrations/pull/start", customer_id=customer_id),
            json={
                "endpoint_url": endpoint,
                "polling_interval_seconds": 0.2,
                "auth_type": "none",
                "run_id": run_id,
                "retry_max_attempts": 1,
                "retry_backoff_seconds": 0.05,
                "request_timeout_seconds": 2.0,
            },
        )
        assert start.status_code == 200
        pulled = _wait_for_pull_ingest(client, customer_id=customer_id, min_count=1)
        assert int(pulled.get("total_ingested", 0)) >= 1

        recent = client.get(_customer_path(f"/results/recent?run_id={run_id}&limit=1", customer_id=customer_id))
        assert recent.status_code == 200
        assert recent.json()["count"] >= 1
        latest = recent.json()["results"][0]
        sensors = latest.get("sensor_relationships") or []
        assert "temperature" in sensors
    finally:
        client.post(_customer_path("/integrations/pull/stop", customer_id=customer_id))
        server.stop()

