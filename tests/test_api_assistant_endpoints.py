from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_client(tmp_path) -> TestClient:
    store = ResultStore(db_path=str(tmp_path / "test_assistant_api.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    service = StructuralMonitoringService(engine=engine, store=store)
    app = create_app(service=service)
    return TestClient(app)


def _q(path: str) -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}customer_id=customer-a&run_id=run-assistant"


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


def _seed(client: TestClient) -> None:
    client.post(_q("/ingest/frame"), json=_frame(0, 50.0))
    client.post(_q("/ingest/frame"), json=_frame(1, 60.0))


def test_assistant_endpoints_available_and_grounded(tmp_path) -> None:
    client = _build_client(tmp_path)
    _seed(client)

    summary = client.post("/assistant/summary", json={"customer_id": "customer-a", "run_id": "run-assistant"})
    handoff = client.post("/assistant/handoff", json={"customer_id": "customer-a", "run_id": "run-assistant"})
    explain = client.post(
        "/assistant/explain",
        json={"customer_id": "customer-a", "run_id": "run-assistant", "mode": "why_recommended"},
    )

    assert summary.status_code == 200
    assert handoff.status_code == 200
    assert explain.status_code == 200

    summary_body = summary.json()
    assert summary_body["mode"] == "summary"
    assert "recommendation" in summary_body["context"]
    assert "explanation" in summary_body["context"]
    assert "memory_recall" in summary_body["context"]
    assert "Observed:" in summary_body["text"]


def test_assistant_explain_rejects_unsupported_mode(tmp_path) -> None:
    client = _build_client(tmp_path)
    _seed(client)

    response = client.post(
        "/assistant/explain",
        json={"customer_id": "customer-a", "run_id": "run-assistant", "mode": "handoff"},
    )

    assert response.status_code == 400
    assert "mode must be one of" in response.json()["detail"]


def test_assistant_report_endpoint_supports_all_modes(tmp_path) -> None:
    client = _build_client(tmp_path)
    _seed(client)

    for mode in ("client_report", "technician_summary", "inspection_brief", "handoff_note"):
        response = client.post(
            "/assistant/report",
            json={"customer_id": "customer-a", "run_id": "run-assistant", "mode": mode, "history_limit": 20},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == mode
        assert isinstance(body["sections"], dict)
        assert isinstance(body["report_text"], str)
        assert body["report_text"]


def test_assistant_report_export_downloads_text(tmp_path) -> None:
    client = _build_client(tmp_path)
    _seed(client)

    response = client.post(
        "/assistant/report/export?format=md",
        json={"customer_id": "customer-a", "run_id": "run-assistant", "mode": "client_report", "history_limit": 20},
    )

    assert response.status_code == 200
    assert "attachment; filename=" in response.headers.get("content-disposition", "")
    assert "Client Report" in response.text
