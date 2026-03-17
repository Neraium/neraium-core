"""Compatibility wrapper for the structural engine."""

from neraium_core.engine import StructuralEngine

__all__ = ["StructuralEngine"]


if __name__ == "__main__":
    engine = StructuralEngine()

    sample_frames = [
        {
            "timestamp": f"2026-03-13T00:{i:02d}:00Z",
            "site_id": "site-a",
            "asset_id": "asset-1",
            "sensor_values": {
                "temp": 50.0 + (i * 0.02),
                "pressure": 100.0 + (i * 0.03),
                "vibration": 1.0 + (i * 0.005),
            },
        }
        for i in range(40)
    ]

    for frame in sample_frames:
        output = engine.process_frame(frame)

    print(output)
