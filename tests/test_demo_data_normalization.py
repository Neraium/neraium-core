from __future__ import annotations

from datetime import datetime, timezone

from ui.demo_data import _normalize_engine_result, _normalize_timestamp


def test_normalize_timestamp_handles_epoch_numeric_strings_without_overflow() -> None:
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert _normalize_timestamp("1712059200000", 0, base_time) == "2024-04-02T12:00:00+00:00"
    assert _normalize_timestamp("1712059200", 0, base_time) == "2024-04-02T12:00:00+00:00"


def test_normalize_engine_result_preserves_explicit_false_signal_emitted() -> None:
    frame = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "site",
        "asset_id": "asset",
        "sensor_values": {"temperature": 32.0},
    }
    result = {
        "state": "instability",
        "signal_emitted": False,
        "structural_drift_score": 0.9,
        "relational_stability_score": 0.1,
    }
    normalized = _normalize_engine_result(result, frame)
    assert normalized["event_admitted"] is False
