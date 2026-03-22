from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _client(tmp_path) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "test_mvp.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service)
    return TestClient(app)


def _run_and_ingest(client: TestClient) -> tuple[str, int]:
    run = client.post(
        "/runs",
        json={"name": "mvp-run", "activate": True, "config": {"baseline_window": 5, "recent_window": 3}},
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["run_id"]
    ing = client.post(
        f"/ingest?run_id={run_id}",
        json={
            "timestamp": "2026-01-01T00:00:00+00:00",
            "site_id": "site-a",
            "asset_id": "asset-a",
            "sensor_values": {"pressure": 51.2, "flow": 12.4},
        },
    )
    assert ing.status_code == 200
    rid = int(ing.json()["results"][0]["result_id"])
    return run_id, rid


def test_mvp_routes_available(tmp_path) -> None:
    client = _client(tmp_path)
    home = client.get("/")
    assert home.status_code == 200
    assert "text/html" in home.headers.get("content-type", "")
    js = client.get("/web/app.js")
    css = client.get("/web/styles.css")
    assert js.status_code == 200
    assert css.status_code == 200


def test_run_scoped_result_detail_and_recent(tmp_path) -> None:
    client = _client(tmp_path)
    run_id, result_id = _run_and_ingest(client)

    detail = client.get(f"/results/{result_id}?run_id={run_id}")
    assert detail.status_code == 200
    result = detail.json()["result"]
    assert result["run_id"] == run_id
    assert result["result_id"] == result_id

    recent = client.get(f"/results/recent?run_id={run_id}&limit=10")
    assert recent.status_code == 200
    body = recent.json()
    assert body["count"] >= 1
    assert body["results"][0]["run_id"] == run_id


def test_export_json_and_csv(tmp_path) -> None:
    client = _client(tmp_path)
    run_id, _ = _run_and_ingest(client)

    export_json = client.get(f"/results/export?run_id={run_id}&format=json&limit=50")
    assert export_json.status_code == 200
    j = export_json.json()
    assert j["format"] == "json"
    decoded = json.loads(j["content"])
    assert isinstance(decoded, list)
    assert decoded

    export_csv = client.get(f"/results/export?run_id={run_id}&format=csv&limit=50")
    assert export_csv.status_code == 200
    c = export_csv.json()
    assert c["format"] == "csv"
    assert "result_id,run_id,timestamp" in c["content"]


def test_update_and_activate_run_routes(tmp_path) -> None:
    client = _client(tmp_path)
    r1 = client.post("/runs", json={"name": "run-1", "activate": True, "config": {"x": 1}})
    r2 = client.post("/runs", json={"name": "run-2", "activate": False, "config": {"x": 2}})
    assert r1.status_code == 200
    assert r2.status_code == 200
    run2 = r2.json()["run"]["run_id"]

    upd = client.patch(f"/runs/{run2}", json={"name": "run-2b", "status": "open", "config": {"x": 3}})
    assert upd.status_code == 200
    assert upd.json()["run"]["name"] == "run-2b"

    act = client.post(f"/runs/{run2}/activate")
    assert act.status_code == 200
    assert act.json()["run"]["is_active"] is True

    active = client.get("/runs/active")
    assert active.status_code == 200
    assert active.json()["run"]["run_id"] == run2

