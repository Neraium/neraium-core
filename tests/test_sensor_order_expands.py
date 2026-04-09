"""Sensor channel list must grow when new keys appear (not frozen to first frame)."""

from __future__ import annotations

import math

from neraium_core.alignment import StructuralEngine


def test_sensor_order_merges_new_keys_across_frames() -> None:
    eng = StructuralEngine(baseline_window=10, recent_window=4, window_stride=1)
    base = {"timestamp": "2025-01-01T00:00:00Z", "site_id": "s", "asset_id": "a"}
    for i in range(5):
        eng.process_frame(
            {
                **base,
                "sensor_values": {"a": float(i), "b": float(i + 1), "c": float(i + 2), "d": float(i + 3)},
            }
        )
    assert list(eng.sensor_order) == ["a", "b", "c", "d"]
    eng.process_frame(
        {
            **base,
            "sensor_values": {
                "a": 1.0,
                "b": 1.0,
                "c": 1.0,
                "d": 1.0,
                "e": 9.0,
                "f": 10.0,
            },
        }
    )
    assert list(eng.sensor_order) == ["a", "b", "c", "d", "e", "f"]
    assert len(eng.frames) == 6
    assert eng.frames[-1]["_vector"].shape[0] == 6


def test_sensor_order_registers_only_real_numeric_channels() -> None:
    eng = StructuralEngine(baseline_window=6, recent_window=3, window_stride=1)
    base = {"timestamp": "2025-01-01T00:00:00Z", "site_id": "s", "asset_id": "a"}

    eng.process_frame(
        {
            **base,
            "sensor_values": {
                "a": 1.0,
                "b": "2.0",
                "bad_text": "n/a",
                "bad_bool": True,
                "bad_inf": math.inf,
            },
        }
    )

    assert list(eng.sensor_order) == ["a", "b"]
