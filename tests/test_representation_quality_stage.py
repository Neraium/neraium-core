from __future__ import annotations

from dataclasses import fields

import numpy as np

from neraium_core.alignment import StructuralEngine, _to_epoch_seconds
from neraium_core.context_invariant_representation import build_temporal_representation
from neraium_core.data_quality import (
    compute_data_quality,
    data_quality_summary,
    impute_missing_simple,
    should_use_degraded_analytics,
)
from neraium_core.engine_stages.ingress_history import IngressAndHistoryBuffersInput, prepare_ingress_and_history_buffers
from neraium_core.engine_stages.representation_quality import (
    RepresentationAndQualityInput,
    RepresentationAndQualityResult,
    build_representation_and_quality,
)
from neraium_core.geometry import normalize_window


def _frame(ts: object, sensors: dict[str, object]) -> dict[str, object]:
    return {
        "timestamp": ts,
        "site_id": "site-a",
        "asset_id": "asset-a",
        "customer_id": "cust-a",
        "sensor_values": sensors,
    }


def _seed_history(engine: StructuralEngine, samples: list[dict[str, object]]) -> None:
    for sample in samples:
        vector = engine._vector_from_frame(sample)
        prepare_ingress_and_history_buffers(
            engine,
            IngressAndHistoryBuffersInput(frame=sample, vector=vector),
            to_epoch_seconds=_to_epoch_seconds,
            incremental_windows_enabled=True,
        )


def test_representation_quality_stage_contract_and_shapes() -> None:
    engine = StructuralEngine(baseline_window=4, recent_window=3)
    _seed_history(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 5.0}),
            _frame(2.0, {"a": 2.0, "b": 4.0}),
            _frame(3.0, {"a": 3.0, "b": 3.0}),
            _frame(4.0, {"a": 4.0, "b": 2.0}),
        ],
    )

    result = build_representation_and_quality(
        engine,
        RepresentationAndQualityInput(
            history_matrix=engine._history_ring.chronological_matrix(),
            history_timestamps=engine._history_ring.chronological_timestamps(),
        ),
    )

    assert [f.name for f in fields(RepresentationAndQualityInput)] == ["history_matrix", "history_timestamps"]
    assert isinstance(result, RepresentationAndQualityResult)
    assert result.baseline_window.shape[0] == 4
    assert result.recent_window.shape[0] == 3
    assert result.z_baseline.shape == result.baseline_window.shape
    assert result.z_recent.shape == result.recent_window.shape


def test_representation_quality_stage_handles_missing_invalid_data_with_degraded_path() -> None:
    engine = StructuralEngine(baseline_window=4, recent_window=3)
    _seed_history(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 2.0}),
            _frame(2.0, {"a": None, "b": 2.1}),
            _frame(3.0, {"a": "invalid", "b": 2.2}),
            _frame(4.0, {"a": 4.0, "b": None}),
        ],
    )

    result = build_representation_and_quality(
        engine,
        RepresentationAndQualityInput(
            history_matrix=engine._history_ring.chronological_matrix(),
            history_timestamps=engine._history_ring.chronological_timestamps(),
        ),
    )

    assert not bool(result.data_quality_report.gate_passed)
    assert isinstance(result.dq_summary.get("statuses"), list)
    assert np.isfinite(np.nan_to_num(result.recent_window)).all()


def test_representation_quality_stage_reports_data_quality_flags() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    _seed_history(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 10.0}),
            _frame(2.0, {"a": 2.0, "b": 10.0}),
            _frame(3.0, {"a": 3.0, "b": 10.0}),
            _frame(4.0, {"a": 4.0, "b": 10.0}),
            _frame(5.0, {"a": 5.0, "b": 10.0}),
        ],
    )
    result = build_representation_and_quality(
        engine,
        RepresentationAndQualityInput(
            history_matrix=engine._history_ring.chronological_matrix(),
            history_timestamps=engine._history_ring.chronological_timestamps(),
        ),
    )

    statuses = set(result.dq_summary.get("statuses", []))
    assert "LOW_SENSOR_COVERAGE" in statuses or "INSUFFICIENT_VARIABILITY" in statuses


def test_representation_quality_stage_compatible_with_ingress_output() -> None:
    engine = StructuralEngine(baseline_window=4, recent_window=3)
    first = _frame(1.0, {"a": 1.0, "b": 2.0})
    vector = engine._vector_from_frame(first)
    prepare_ingress_and_history_buffers(
        engine,
        IngressAndHistoryBuffersInput(frame=first, vector=vector),
        to_epoch_seconds=_to_epoch_seconds,
        incremental_windows_enabled=True,
    )
    for ts in (2.0, 3.0, 4.0):
        engine.process_frame(_frame(ts, {"a": ts, "b": ts + 1.0}))

    result = build_representation_and_quality(
        engine,
        RepresentationAndQualityInput(
            history_matrix=engine._history_ring.chronological_matrix(),
            history_timestamps=engine._history_ring.chronological_timestamps(),
        ),
    )

    assert result.ts_recent is not None
    assert len(result.ts_recent) >= 2


def test_representation_quality_stage_preserves_downstream_scoring_inputs() -> None:
    engine = StructuralEngine(baseline_window=4, recent_window=3)
    _seed_history(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 2.0}),
            _frame(2.0, {"a": 2.0, "b": 3.0}),
            _frame(3.0, {"a": 3.0, "b": 4.0}),
            _frame(4.0, {"a": 4.0, "b": 5.0}),
        ],
    )

    history_matrix = engine._history_ring.chronological_matrix()
    history_ts = engine._history_ring.chronological_timestamps()
    stage_result = build_representation_and_quality(
        engine,
        RepresentationAndQualityInput(history_matrix=history_matrix, history_timestamps=history_ts),
    )

    rep = build_temporal_representation(history_matrix, engine.representation_config, timestamps=history_ts)
    transformed_history = rep.transformed
    expected_baseline = np.asarray(transformed_history[: engine.baseline_window][:: engine.window_stride], dtype=float)
    expected_recent = np.asarray(transformed_history[-engine.recent_window :][:: engine.window_stride], dtype=float)
    expected_dq = compute_data_quality(
        expected_baseline,
        expected_recent,
        sensor_names=engine.sensor_order,
        timestamps_baseline=engine._get_baseline_timestamps(None),
        timestamps_recent=engine._get_recent_timestamps(None),
    )
    if not expected_dq.gate_passed and should_use_degraded_analytics(expected_dq):
        expected_baseline = impute_missing_simple(expected_baseline, method="column_mean")
        expected_recent = impute_missing_simple(expected_recent, method="column_mean")
    expected_zb, _, _ = normalize_window(expected_baseline)
    expected_zr, _, _ = normalize_window(expected_recent)

    assert np.allclose(stage_result.baseline_window, expected_baseline, equal_nan=True)
    assert np.allclose(stage_result.recent_window, expected_recent, equal_nan=True)
    assert np.allclose(stage_result.z_baseline, expected_zb, equal_nan=True)
    assert np.allclose(stage_result.z_recent, expected_zr, equal_nan=True)
    assert stage_result.dq_summary == data_quality_summary(expected_dq)
