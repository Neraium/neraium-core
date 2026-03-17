from neraium_core.service import StructuralIntelligenceService


def test_service_ingest_valid_payload_returns_result() -> None:
    service = StructuralIntelligenceService()
    result = service.ingest_frame(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site",
            "asset_id": "asset",
            "sensor_values": {"temp": 1.0, "p": 2.0},
        }
    )
    assert result.timestamp == "2026-01-01T00:00:00Z"


def test_service_ingest_csv_works() -> None:
    service = StructuralIntelligenceService()
    response = service.ingest_csv(
        "timestamp,site_id,asset_id,temp,p\n"
        "2026-01-01T00:00:00Z,s1,a1,1.0,2.0\n"
        "2026-01-01T00:01:00Z,s1,a1,1.2,2.2\n"
    )
    assert response.summary.accepted == 2


def test_latest_result_after_ingest() -> None:
    service = StructuralIntelligenceService()
    service.ingest_frame(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site",
            "asset_id": "asset",
            "sensor_values": {"temp": 1.0, "p": 2.0},
        }
    )
    assert service.latest_result() is not None


def test_service_reset() -> None:
    service = StructuralIntelligenceService()
    service.ingest_frame(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "site",
            "asset_id": "asset",
            "sensor_values": {"temp": 1.0, "p": 2.0},
        }
    )
    service.reset()
    assert service.latest_result() is None
