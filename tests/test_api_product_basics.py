from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from apps.api.main import create_app, is_api_key_valid
from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _build_service(tmp_path) -> StructuralMonitoringService:
    store = ResultStore(db_path=str(tmp_path / "test.db"))
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    return StructuralMonitoringService(engine=engine, store=store)


def _build_client(tmp_path) -> TestClient:
    app = create_app(service=_build_service(tmp_path))
    return TestClient(app)


def _customer_path(path: str, customer_id: str = "customer-a") -> str:
    sep = "&" if "?" in path else "?"
    return f"{path}{sep}customer_id={customer_id}"


def test_api_key_rejected_when_configured() -> None:
    assert is_api_key_valid("secret", None) is False
    assert is_api_key_valid("secret", "wrong") is False


def test_api_key_accepted_when_configured() -> None:
    assert is_api_key_valid("secret", "secret") is True


def test_service_persistence_and_reset_interaction(tmp_path) -> None:
    service = _build_service(tmp_path)

    for i in range(6):
        result = service.ingest_payload(
            {
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "site_id": "s1",
                "asset_id": "a1",
                "sensor_values": {"pressure": 50.0 + i},
            }
        )
        assert result["interpretation"]["heuristic"] is True

    latest = service.get_latest_result()
    assert latest is not None

    recent = service.list_recent_results(limit=3)
    assert len(recent) == 3

    service.reset()

    assert service.get_latest_result() is None


def test_api_endpoints_registered(tmp_path) -> None:
    app = create_app(service=_build_service(tmp_path))
    paths = {route.path for route in app.router.routes}

    assert "/health" in paths
    assert "/client-errors" in paths
    assert "/ingest" in paths
    assert "/ingest/frame" in paths
    assert "/ingest/batch" in paths
    assert "/ingest/csv" in paths
    assert "/ingest/csv/preview" in paths
    assert "/reset" in paths
    assert "/results/latest" in paths
    assert "/results/recent" in paths
    assert "/state" in paths
    assert "/history" in paths
    assert "/recommendation" in paths
    assert "/recommendations/latest" in paths
    assert "/decision" in paths
    assert "/explanation" in paths
    assert "/events/latest" in paths


def test_operator_fields_present_and_structural_skip_for_single_signal(tmp_path) -> None:
    service = _build_service(tmp_path)

    result = service.ingest_payload(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "site_id": "s1",
            "asset_id": "a1",
            "sensor_values": {"pressure": 50.0},
        }
    )

    assert result["risk_level"] == "LOW"
    assert result["action_state"] == "STABLE"
    assert isinstance(result["operator_message"], str)
    assert result["trend"] in {"UNKNOWN", "STABLE", "RISING", "FALLING"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["structural_analysis_available"] is False
    assert result["skipped_reason"] == "insufficient signal dimensionality"


def test_health_includes_runtime_configuration_flags(tmp_path) -> None:
    client = _build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["latest_result_available"] is False
    assert "auth_configured" in body
    assert "persistence_available" in body


def test_results_endpoints_return_stable_shape(tmp_path) -> None:
    client = _build_client(tmp_path)

    ingest_response = client.post(
        _customer_path("/ingest", customer_id="customer-a"),
        json={
            "timestamp": "2026-01-01T00:00:00+00:00",
            "site_id": "s1",
            "asset_id": "a1",
            "sensor_values": {"pressure": 50.0},
        },
    )
    assert ingest_response.status_code == 200

    latest_response = client.get(_customer_path("/results/latest", customer_id="customer-a"))
    recent_response = client.get(_customer_path("/results/recent?limit=5", customer_id="customer-a"))

    core_keys = {"latest", "count", "results"}
    for response in (latest_response, recent_response):
        body = response.json()
        assert core_keys.issubset(body.keys())
        assert isinstance(body["results"], list)
        assert body["count"] == len(body["results"])
    ingest_body = ingest_response.json()
    assert core_keys.issubset(ingest_body.keys())
    assert isinstance(ingest_body["results"], list)
    assert ingest_body["count"] == len(ingest_body["results"])


def test_recent_results_ordering_and_limit(tmp_path) -> None:
    client = _build_client(tmp_path)

    for i in range(5):
        resp = client.post(
            _customer_path("/ingest", customer_id="customer-a"),
            json={
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "site_id": "s1",
                "asset_id": "a1",
                "sensor_values": {"pressure": 50.0 + i},
            },
        )
        assert resp.status_code == 200

    recent = client.get(_customer_path("/results/recent?limit=3", customer_id="customer-a"))
    assert recent.status_code == 200

    body = recent.json()
    assert body["count"] == 3
    timestamps = [item["timestamp"] for item in body["results"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_onboarding_flow_bootstraps_run_and_test_frame(tmp_path) -> None:
    client = _build_client(tmp_path)
    customer_id = "onboarding-customer"

    verify = client.post(
        "/onboarding/verify",
        json={"customer_id": customer_id, "api_key": "dev-key", "environment_label": "pilot"},
    )
    assert verify.status_code == 200
    verify_body = verify.json()
    assert verify_body["credentials_verified"] is True
    assert "api_key" not in verify_body

    profile = client.post(
        _customer_path("/onboarding/profile", customer_id=customer_id),
        json={
            "customer_display_name": "Acme Manufacturing",
            "site_name": "Site A",
            "default_asset_id": "pump-01",
            "asset_group": "north-line",
            "ingestion_method": "api_push",
        },
    )
    assert profile.status_code == 200

    bootstrap = client.post(_customer_path("/onboarding/bootstrap-run", customer_id=customer_id))
    assert bootstrap.status_code == 200
    run_id = bootstrap.json()["run"]["run_id"]
    assert run_id

    test_frame = client.post(_customer_path("/onboarding/test-frame", customer_id=customer_id))
    assert test_frame.status_code == 200
    body = test_frame.json()
    assert body["ingest_accepted"] is True
    assert body["active_run_id"] == run_id

    status = client.get(_customer_path("/onboarding/status", customer_id=customer_id))
    assert status.status_code == 200
    status_body = status.json()
    assert status_body["ready"] is True
    assert status_body["test_frame_sent"] is True


def test_onboarding_verify_rejects_invalid_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_API_KEY", "expected-secret")
    service = _build_service(tmp_path)
    with TestClient(create_app(service=service)) as client:
        response = client.post(
            "/onboarding/verify",
            json={"customer_id": "cust-x", "api_key": "wrong"},
        )
    assert response.status_code == 401
