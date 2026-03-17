from fastapi import HTTPException

from apps.api.main import ingest, ingest_batch, reset, result, service

VALID_PAYLOAD = {
    "timestamp": "2024-01-01T00:00:00Z",
    "site_id": "site-1",
    "asset_id": "asset-1",
    "sensor_values": {"pressure": 80.0, "flow": 120.0},
}


def test_ingest_returns_latest_result() -> None:
    service.reset()
    response = ingest(VALID_PAYLOAD)

    assert response.site_id == "site-1"
    assert response.asset_id == "asset-1"
    assert response.event_type == "baseline_telemetry"


def test_batch_ingest_returns_latest_result() -> None:
    service.reset()
    payloads = [
        VALID_PAYLOAD,
        {
            **VALID_PAYLOAD,
            "timestamp": "2024-01-01T00:01:00Z",
            "sensor_values": {"pressure": 81.0, "flow": 119.5},
        },
    ]

    response = ingest_batch(payloads)

    assert response.timestamp.startswith("2024-01-01T00:01:00")


def test_result_and_reset() -> None:
    service.reset()
    ingest(VALID_PAYLOAD)

    result_response = result()
    assert result_response.site_id == "site-1"

    reset_response = reset()
    assert reset_response == {"ok": True}

    try:
        result()
        raise AssertionError("Expected HTTPException after reset")
    except HTTPException as exc:
        assert exc.status_code == 404


def test_invalid_payload_returns_http_400() -> None:
    service.reset()

    try:
        ingest({"sensor_values": []})
        raise AssertionError("Expected HTTPException for invalid payload")
    except HTTPException as exc:
        assert exc.status_code == 400
