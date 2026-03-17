import pytest

pytest.importorskip("numpy")

from neraium_core.engine import StructuralEngine


def test_baseline_and_latest_result() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=3)

    for i in range(8):
        result = engine.process_frame(
            {
                "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
                "site_id": "site",
                "asset_id": "asset",
                "sensor_values": {"a": 1.0 + (i * 0.01), "b": 2.0 + (i * 0.01)},
            }
        )

    assert result["structural_drift_score"] >= 0.0
    assert engine.get_latest_result() == result


def test_reset_clears_engine_state() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    engine.process_frame(
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "site_id": "site",
            "asset_id": "asset",
            "sensor_values": {"a": 1.0},
        }
    )
    assert len(engine.frames) == 1
    engine.reset()
    assert len(engine.frames) == 0
    assert engine.sensor_order == []
    assert engine.get_latest_result() is None
