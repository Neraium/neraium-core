from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_service(tmp_path) -> StructuralMonitoringService:
    store = ResultStore(db_path=str(tmp_path / "test.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    return StructuralMonitoringService(engine=engine, store=store)


def test_client_errors_post_returns_204(tmp_path) -> None:
    app = create_app(service=_build_service(tmp_path))
    client = TestClient(app)
    res = client.post(
        "/client-errors",
        json={
            "message": "test error",
            "stack": "Error: test\n  at x (app.js:1:1)",
            "url": "http://test/run",
            "reason": "unit_test",
        },
    )
    assert res.status_code == 204
    assert res.content == b""


def test_client_errors_accepts_minimal_body(tmp_path) -> None:
    app = create_app(service=_build_service(tmp_path))
    client = TestClient(app)
    res = client.post("/client-errors", json={})
    assert res.status_code == 204
