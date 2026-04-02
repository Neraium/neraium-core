from __future__ import annotations

from neraium_core.alignment import StructuralEngine


def test_single_sensor_stream_is_stable_and_skips_relational_metrics() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)

    outputs = []
    for index in range(10):
        frame = {
            "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
            "site_id": "site-a",
            "asset_id": "asset-1",
            "sensor_values": {"pressure": 60.0 + (index * 0.2)},
        }
        outputs.append(engine.process_frame(frame))

    latest = outputs[-1]
    assert latest["structural_drift_score"] >= 0.0
    assert "experimental_analytics" in latest
    assert latest["experimental_analytics"]["relational_metrics_skipped"] is True
    assert "directional" not in latest["experimental_analytics"]
    assert "subsystems" not in latest["experimental_analytics"]
    assert "early_warning" in latest["experimental_analytics"]
