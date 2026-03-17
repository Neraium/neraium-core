from neraium_core.service import StructuralIntelligenceService


if __name__ == "__main__":
    service = StructuralIntelligenceService()
    for i in range(40):
        payload = {
            "timestamp": f"2026-03-13T00:{i:02d}:00Z",
            "site_id": "site-a",
            "asset_id": "asset-1",
            "sensor_values": {
                "temp": 50.0 + (i * 0.02),
                "pressure": 100.0 + (i * 0.03),
                "vibration": 1.0 + (i * 0.005),
            },
        }
        result = service.ingest_frame(payload)

    print(result.to_dict())
