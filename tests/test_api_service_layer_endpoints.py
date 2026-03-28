from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.output_contract import CANONICAL_SCHEMA_VERSION
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_client(tmp_path) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "test_service_api.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service)
    return TestClient(app)


def _q(path: str) -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}customer_id=customer-a&run_id=run-prod"


def _frame(i: int, pressure: float) -> dict[str, object]:
    return {
        "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
        "site_id": "plant-1",
        "asset_id": "pump-9",
        "sensor_values": {
            "pressure": pressure,
            "temperature": 80.0 + i,
        },
    }


def test_ingest_frame_endpoint_returns_canonical_payload(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(_q("/ingest/frame"), json=_frame(0, 50.0))

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == CANONICAL_SCHEMA_VERSION
    assert body["cycle"] == 1
    assert isinstance(body["decision"], dict)


def test_state_and_history_endpoints_return_service_history(tmp_path) -> None:
    client = _build_client(tmp_path)
    client.post(_q("/ingest/frame"), json=_frame(0, 50.0))
    client.post(_q("/ingest/frame"), json=_frame(1, 55.0))

    state_response = client.get(_q("/state"))
    history_response = client.get(_q("/history?limit=2"))

    assert state_response.status_code == 200
    state = state_response.json()["state"]
    assert state is not None
    assert state["cycle"] == 2

    assert history_response.status_code == 200
    history_body = history_response.json()
    assert history_body["count"] == 2
    assert len(history_body["history"]) == 2


def test_decision_and_explanation_endpoints_return_latest_values(tmp_path) -> None:
    client = _build_client(tmp_path)
    client.post(_q("/ingest/frame"), json=_frame(0, 50.0))

    decision_response = client.get(_q("/decision"))
    explanation_response = client.get(_q("/explanation"))

    assert decision_response.status_code == 200
    assert isinstance(decision_response.json()["decision"], dict)
    assert "state" in decision_response.json()["decision"]

    assert explanation_response.status_code == 200
    assert isinstance(explanation_response.json()["explanation_text"], str)


def test_events_latest_endpoint_returns_latest_event_flags(tmp_path) -> None:
    client = _build_client(tmp_path)
    client.post(_q("/ingest/frame"), json=_frame(0, 50.0))

    response = client.get(_q("/events/latest"))

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["events"], list)
    assert "decision_available" in body["events"]
    assert body["cycle"] == 1
