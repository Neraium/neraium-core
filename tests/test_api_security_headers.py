from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_client(tmp_path) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "test_security_headers.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service)
    return TestClient(app)


def test_default_security_headers_present_on_health_response(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert response.headers.get("permissions-policy") == "camera=(), microphone=(), geolocation=()"
    assert response.headers.get("cache-control") == "no-store"


def test_existing_cache_control_is_preserved_for_static_assets(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/web/app.js")

    assert response.status_code == 200
    assert "public" in response.headers.get("cache-control", "")
