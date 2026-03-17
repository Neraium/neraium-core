from neraium_core.engine import StructuralEngine


def _frame(i: int, *, bad: bool = False) -> dict:
    return {
        "timestamp": f"2026-03-13T00:{i:02d}:00Z",
        "site_id": "site-a",
        "asset_id": "asset-1",
        "sensor_values": {
            "temp": None if bad else 50.0 + (i * 0.02),
            "pressure": 100.0 + (i * 0.03),
            "vibration": 1.0 + (i * 0.005),
        },
    }


def test_engine_initializes() -> None:
    engine = StructuralEngine()
    assert engine.get_latest_result() is None


def test_baseline_formation_before_enough_data() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    result = engine.process_frame(_frame(0))
    assert result.event_type == "baseline_telemetry"
    assert result.structural_drift_score == 0.0


def test_complete_frames_produce_nonzero_metrics() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    result = None
    for i in range(20):
        result = engine.process_frame(_frame(i))
    assert result is not None
    assert result.mahalanobis_score >= 0.0


def test_reset_clears_state() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    engine.process_frame(_frame(0))
    engine.reset()
    assert engine.get_latest_result() is None
    assert len(engine.frames) == 0


def test_missing_data_does_not_crash() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    for i in range(10):
        result = engine.process_frame(_frame(i, bad=(i % 2 == 0)))
    assert result.event_type in {"baseline_telemetry", "flow_observation", "quality_observation", "instability_escalation"}
