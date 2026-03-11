from datetime import datetime, timezone

from neraium_core.alignment import AlignmentEngine
from neraium_core.models import SystemDefinition, TelemetryPayload


def test_alignment_basic():

    definition = SystemDefinition(
        system_id="sys_1",
        schema_version="1.0",
        raw_sample_period_seconds=1,
        inference_window_seconds=5,
        max_forward_fill_windows=1,
        max_missing_signal_fraction=1.0,
        signals=[
            {"name": "a", "dtype": "float64", "unit": "x", "required_for_scoring": True},
            {"name": "b", "dtype": "int64", "unit": "x", "required_for_scoring": True},
        ],
        vector_order=["a", "b"],
    )

    engine = AlignmentEngine(definition)

    payload = TelemetryPayload(
        system_id="sys_1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        signals={"a": 1.0, "b": 2},
    )

    windows = engine.ingest(payload)

    assert isinstance(windows, list)
