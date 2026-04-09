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


def test_historical_csv_options_and_source(tmp_path) -> None:
    app = create_app(service=_build_service(tmp_path))
    client = TestClient(app)

    options = client.get("/demo/historical/csv-options")
    assert options.status_code == 200
    payload = options.json()
    assert isinstance(payload.get("sources"), list)
    assert payload["sources"]

    first_key = str(payload["sources"][0]["key"])
    source = client.get("/demo/historical/csv-source", params={"source_key": first_key})
    assert source.status_code == 200
    body = source.json()
    assert body["key"] == first_key
    assert isinstance(body.get("csv_text"), str)
    assert "timestamp" in body["csv_text"]
