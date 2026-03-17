import math

from neraium_core.engine import StructuralEngine

_USE_DEFAULT = object()


def frame(i: int, value: float | None | object = _USE_DEFAULT):
    return {
        "timestamp": f"2026-01-01T00:{i:02d}:00Z",
        "site_id": "site",
        "asset_id": "asset",
        "sensor_values": {
            "a": 1.0 + i * 0.01 if value is _USE_DEFAULT else value,
            "b": 2.0 + i * 0.02,
        },
    }


def test_baseline_formation_behavior():
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    outputs = [engine.process_frame(frame(i)) for i in range(7)]

    assert outputs[0]["event_type"] == "baseline_telemetry"
    assert outputs[-1]["state"] in {"STABLE", "WATCH", "ALERT"}
    assert outputs[-1]["event_type"] != "baseline_telemetry"


def test_none_value_becomes_nan_and_keeps_baseline_output():
    engine = StructuralEngine(baseline_window=6, recent_window=3)
    for i in range(6):
        engine.process_frame(frame(i))

    result = engine.process_frame(frame(7, value=None))
    assert result["event_type"] == "baseline_telemetry"
    assert math.isnan(engine.frames[-1]["_vector"][0])


def test_reset_clears_state():
    engine = StructuralEngine()
    engine.process_frame(frame(1))
    assert engine.get_latest_result() is not None
    engine.reset()
    assert engine.get_latest_result() is None
    assert len(engine.frames) == 0
    assert engine.sensor_order == []
