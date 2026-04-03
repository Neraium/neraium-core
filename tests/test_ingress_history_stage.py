from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone

from neraium_core.alignment import StructuralEngine, _to_epoch_seconds
from neraium_core.engine_stages.ingress_history import (
    IngressAndHistoryBuffersInput,
    IngressAndHistoryBuffersResult,
    prepare_ingress_and_history_buffers,
)


def _frame(ts: object, sensors: dict[str, float]) -> dict[str, object]:
    return {
        "timestamp": ts,
        "site_id": "site-a",
        "asset_id": "asset-a",
        "customer_id": "cust-a",
        "sensor_values": sensors,
    }


def test_ingress_history_stage_contract_shape(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_INCREMENTAL", "1")
    engine = StructuralEngine(baseline_window=4, recent_window=3)
    frame = _frame(1.0, {"s1": 1.0, "s2": 2.0})
    vector = engine._vector_from_frame(frame)

    stage_input = IngressAndHistoryBuffersInput(frame=frame, vector=vector)
    result = prepare_ingress_and_history_buffers(
        engine,
        stage_input,
        to_epoch_seconds=_to_epoch_seconds,
        incremental_windows_enabled=True,
    )

    assert [f.name for f in fields(IngressAndHistoryBuffersInput)] == ["frame", "vector"]
    assert [f.name for f in fields(IngressAndHistoryBuffersResult)] == [
        "history_transition_len_before",
        "history_shock_len_before",
        "history_drift_len_before",
        "ts_val",
        "ts_ring",
    ]
    assert isinstance(result, IngressAndHistoryBuffersResult)
    assert len(engine.frames) == 1
    assert len(engine._history_ring) == 1


def test_warmup_prescore_behavior_unchanged() -> None:
    engine = StructuralEngine(baseline_window=8, recent_window=4)
    out = engine.process_frame(_frame("2026-01-01T00:00:00Z", {"a": 1.0, "b": 2.0}))

    assert out["state"] == "STABLE"
    assert out["structural_drift_score"] == 0.0
    assert out["transition_state"] == "NONE"
    assert out["explanation"] == "Warmup: awaiting sufficient window history."


def test_history_buffers_update_during_prescore_path(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_INCREMENTAL", "1")
    engine = StructuralEngine(baseline_window=20, recent_window=10)
    engine.process_frame(_frame(10.0, {"a": 1.0, "b": 2.0}))
    engine.process_frame(_frame(11.0, {"a": 1.1, "b": 2.1}))

    assert len(engine.frames) == 2
    assert len(engine._history_ring) == 2
    assert len(engine._recent_vector_buffer) == 2
    assert len(engine._recent_ts_buffer) == 2


def test_timestamp_handling_compatibility(monkeypatch) -> None:
    monkeypatch.setenv("NERAIUM_INCREMENTAL", "1")
    engine = StructuralEngine(baseline_window=20, recent_window=10)
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    vector_1 = engine._vector_from_frame(_frame(dt, {"a": 1.0}))
    prepare_ingress_and_history_buffers(
        engine,
        IngressAndHistoryBuffersInput(frame=_frame(dt, {"a": 1.0}), vector=vector_1),
        to_epoch_seconds=_to_epoch_seconds,
        incremental_windows_enabled=True,
    )
    vector_2 = engine._vector_from_frame(_frame("bad-timestamp", {"a": 1.2}))
    result_2 = prepare_ingress_and_history_buffers(
        engine,
        IngressAndHistoryBuffersInput(frame=_frame("bad-timestamp", {"a": 1.2}), vector=vector_2),
        to_epoch_seconds=_to_epoch_seconds,
        incremental_windows_enabled=True,
    )

    ring_ts = engine._history_ring.chronological_timestamps()
    assert ring_ts.shape[0] == 2
    assert float(ring_ts[0]) == float(dt.timestamp())
    assert float(ring_ts[1]) == 1.0
    assert result_2.ts_val == 0.0
    assert engine._recent_ts_buffer.to_list()[-1] == 0.0


def test_process_stream_frame_runtime_compatibility() -> None:
    engine = StructuralEngine(baseline_window=8, recent_window=4)
    out = engine.process_stream_frame(_frame(1.0, {"a": 1.0, "b": 2.0}))

    assert out["site_id"] == "site-a"
    assert out["asset_id"] == "asset-a"
    assert out["engine_ready"] is False
