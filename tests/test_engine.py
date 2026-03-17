import pytest

from neraium_core.engine import StructuralEngine


def _frame(value: float):
    return {
        "timestamp": "2026-01-01T00:00:00Z",
        "site_id": "site-a",
        "asset_id": "asset-a",
        "sensor_values": {"a": value, "b": value + 0.5},
    }


def test_engine_initialization():
    engine = StructuralEngine(baseline_window=10, recent_window=4)
    assert engine.baseline_window == 10
    assert engine.recent_window == 4


def test_ingest_and_score_warmup_then_drift():
    engine = StructuralEngine(baseline_window=8, recent_window=3)

    for _ in range(8):
        engine.ingest(_frame(1.0))
    assert engine.score() >= 0.0

    engine.ingest(_frame(4.0))
    assert engine.score() > 0.0


def test_reset_behavior():
    engine = StructuralEngine(baseline_window=6, recent_window=2)
    for idx in range(6):
        engine.ingest(_frame(float(idx)))

    assert engine.score() >= 0.0
    engine.reset()
    assert engine.score() == 0.0


def test_invalid_config():
    with pytest.raises(ValueError):
        StructuralEngine(baseline_window=5, recent_window=5)
