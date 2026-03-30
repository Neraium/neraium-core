from __future__ import annotations

import pytest

pytest.importorskip("httpx")

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


def _customer_path(path: str, customer_id: str = "customer-a") -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}customer_id={customer_id}"


def test_ingest_invalid_timestamp_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        _customer_path("/ingest"),
        json={
            "timestamp": "not-a-real-timestamp",
            "site_id": "s1",
            "asset_id": "a1",
            "sensor_values": {"pressure": 12.3},
        },
    )

    assert response.status_code == 400
    assert "Invalid timestamp" in str(response.json())


def test_ingest_malformed_payload_returns_422(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        _customer_path("/ingest"),
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
    assert "sensor_values" in str(response.json())


def test_ingest_batch_invalid_timestamp_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        _customer_path("/ingest/batch"),
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
    assert "Invalid timestamp" in str(response.json())


def test_ingest_batch_unhandled_error_returns_controlled_json(tmp_path, monkeypatch) -> None:
    client = _build_client(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated ingest failure")

    monkeypatch.setattr("apps.api.main.service_instance.ingest_batch", _boom)

    response = client.post(
        _customer_path("/ingest/batch"),
        json={
            "items": [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "site_id": "s1",
                    "asset_id": "a1",
                    "sensor_values": {"pressure": 10},
                }
            ]
        },
    )

    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error"
    assert body["message"] == "ingest failed"
    assert "simulated ingest failure" in body["detail"]


def test_ingest_csv_malformed_payload_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.post(
        _customer_path("/ingest/csv"),
        json={"csv_text": "timestamp,site_id,asset_id,s1\ninvalid,s,a,1"},
    )

    assert response.status_code == 400
    assert response.status_code != 500
    assert "Invalid CSV row" in str(response.json())


def test_ingest_accepts_alias_signal_payload(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest"),
        json={
            "recorded_at": "2026-01-01T00:00:00Z",
            "entity_id": "a1",
            "location": "s1",
            "variables": {"pressure": 12.3},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["results"][0]["asset_id"] == "a1"


def test_ingest_batch_accepts_stream_records_shape(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest/batch"),
        json={
            "stream": {
                "records": [
                    {
                        "recorded_at": "2026-01-01T00:00:00Z",
                        "entity_id": "a1",
                        "location": "s1",
                        "variables": {"pressure": 10.0},
                    }
                ]
            }
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_ingest_json_items_shape(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest/json"),
        json={
            "items": [
                {
                    "recorded_at": "2026-01-01T00:00:00Z",
                    "entity_id": "a1",
                    "variables": {"pressure": 10.5},
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_ingest_canonical_direct_path(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest/canonical"),
        json={
            "records": [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "asset_id": "a1",
                    "signals": {"pressure": 9.0},
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_ingest_json_ambiguous_mapping_returns_400(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest/json"),
        json={
            "items": [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "time": "2026-01-01T00:00:01+00:00",
                    "asset_id": "a1",
                    "signals": {"pressure": 9.0},
                }
            ]
        },
    )
    assert response.status_code == 400
    assert "ambiguous_mapping" in str(response.json())


def test_ingest_frame_missing_asset_returns_structured_error(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest/frame"),
        json={
            "timestamp": "2026-01-01T00:00:00+00:00",
            "site_id": "s1",
            "sensor_values": {"pressure": 12.3},
        },
    )
    assert response.status_code in {400, 422}
    body = response.json()
    assert "message" in body or "detail" in body


def test_ingest_batch_partial_success_isolated_failures(tmp_path) -> None:
    client = _build_client(tmp_path)
    response = client.post(
        _customer_path("/ingest/batch"),
        json={
            "items": [
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "site_id": "s1",
                    "asset_id": "a1",
                    "sensor_values": {"pressure": 10},
                },
                {
                    "timestamp": "2026-01-01T00:01:00+00:00",
                    "site_id": "s1",
                    "asset_id": "a1",
                    "sensor_values": {},
                },
            ]
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("status") in {"partial_success", "ok"}
    assert body.get("processed", 0) >= 1
