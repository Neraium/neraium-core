from neraium_core.engine import StructuralEngine


def _frame(i: int) -> dict:
    return {
        "timestamp": f"2026-01-01T00:{i:02d}:00Z",
        "site_id": "site-a",
        "asset_id": "asset-1",
        "sensor_values": {
            "temp": 50.0 + i * 0.1,
            "pressure": 100.0 + i * 0.1,
        },
    }


def test_engine_initializes_correctly() -> None:
    engine = StructuralEngine()

    assert engine.frames.maxlen == 500
    assert engine.sensor_order == []
    assert engine.latest_result is None


def test_engine_processes_frame() -> None:
    engine = StructuralEngine(baseline_window=8, recent_window=4)

    result = None
    for i in range(12):
        result = engine.process_frame(_frame(i))

    assert result is not None
    assert "state" in result
    assert "structural_drift_score" in result


def test_reset_clears_state() -> None:
    engine = StructuralEngine(baseline_window=8, recent_window=4)

    for i in range(12):
        engine.process_frame(_frame(i))

    engine.reset()

    assert len(engine.frames) == 0
    assert engine.sensor_order == []
    assert engine.latest_result is None
    assert engine.prev_drift is None
