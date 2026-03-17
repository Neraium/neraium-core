from __future__ import annotations

import pytest

from neraium_core.service import StructuralIntelligenceService


def test_valid_ingestion() -> None:
    service = StructuralIntelligenceService(baseline_window=2, recent_window=2)

    result = service.ingest_payload(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "s1",
            "asset_id": "a1",
            "sensor_values": {"temp": 10.0, "pressure": 30.0},
        }
    )

    assert result.drift >= 0.0
    assert service.latest_result() is not None


def test_empty_payload_raises() -> None:
    service = StructuralIntelligenceService()

    with pytest.raises(ValueError, match="at least one sensor value"):
        service.ingest_payload({})


def test_partial_data_is_handled() -> None:
    service = StructuralIntelligenceService(baseline_window=2, recent_window=2)

    result = service.ingest_payload(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "sensor_values": {"temp": None, "pressure": 11.2},
        }
    )

    assert result.drift >= 0.0


def test_multiple_frames_batch() -> None:
    service = StructuralIntelligenceService(baseline_window=2, recent_window=2)
    payloads = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "sensor_values": {"temp": 10.0, "pressure": 30.0},
        },
        {
            "timestamp": "2026-01-01T00:01:00Z",
            "sensor_values": {"temp": 11.0, "pressure": 29.5},
        },
    ]

    results = service.ingest_batch(payloads)

    assert len(results) == 2
    assert service.latest_result() == results[-1]


def test_reset_behavior() -> None:
    service = StructuralIntelligenceService()
    service.ingest_payload(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "sensor_values": {"temp": 10.0},
        }
    )

    assert service.latest_result() is not None
    service.reset()
    assert service.latest_result() is None
