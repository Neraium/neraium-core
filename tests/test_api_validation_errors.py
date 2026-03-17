from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_client(tmp_path) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "test_validation_errors.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service)
    return TestClient(app)


def test_ingest_invalid_timestamp_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/ingest",
        json={
            "timestamp": "not-a-real-timestamp",
            "site_id": "s1",
            "asset_id": "a1",
            "sensor_values": {"pressure": 12.3},
        },
    )

    assert response.status_code == 400
    assert "Invalid timestamp" in response.json()["detail"]


def test_ingest_malformed_payload_returns_422(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/ingest",
        json={
            "timestamp": "2026-01-01T00:00:00+00:00",
            "site_id": "s1",
            "asset_id": "a1",
            "sensor_values": "not-an-object",
        },
    )

    # 422 = FastAPI/Pydantic schema validation before handler execution.
    # 400 = service/business validation after request passes schema validation.
    assert response.status_code == 422
    assert response.status_code != 500
    assert "sensor_values" in str(response.json()["detail"])


def test_ingest_batch_invalid_timestamp_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/ingest/batch",
        json={
            "items": [
                {
                    "timestamp": "bad-timestamp",
                    "site_id": "s1",
                    "asset_id": "a1",
                    "sensor_values": {"pressure": 10},
                }
            ]
        },
    )

    assert response.status_code == 400
    assert response.status_code != 500
    assert "Invalid timestamp" in response.json()["detail"]


def test_ingest_csv_malformed_payload_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        "/ingest/csv",
        json={"csv_text": "timestamp,site_id,asset_id,s1\ninvalid,s,a,1"},
    )

    assert response.status_code == 400
    assert response.status_code != 500
    assert "Invalid CSV row" in response.json()["detail"]
