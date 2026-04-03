from __future__ import annotations

from dataclasses import fields

import numpy as np

from neraium_core.alignment import StructuralEngine, _to_epoch_seconds
from neraium_core.engine_stages.ingress_history import IngressAndHistoryBuffersInput, prepare_ingress_and_history_buffers
from neraium_core.engine_stages.representation_quality import RepresentationAndQualityInput, build_representation_and_quality
from neraium_core.engine_stages.scoring_preparation import (
    ScoringPreparationInput,
    ScoringPreparationResult,
    prepare_scoring_inputs,
)
from neraium_core.geometry import correlation_matrix
from neraium_core.staged_pipeline import FeatureExtractionStage


def _frame(ts: object, sensors: dict[str, object]) -> dict[str, object]:
    return {
        "timestamp": ts,
        "site_id": "site-a",
        "asset_id": "asset-a",
        "customer_id": "cust-a",
        "sensor_values": sensors,
    }


def _seed(engine: StructuralEngine, samples: list[dict[str, object]]) -> None:
    for sample in samples:
        vector = engine._vector_from_frame(sample)
        prepare_ingress_and_history_buffers(
            engine,
            IngressAndHistoryBuffersInput(frame=sample, vector=vector),
            to_epoch_seconds=_to_epoch_seconds,
            incremental_windows_enabled=True,
        )


def _representation(engine: StructuralEngine):
    return build_representation_and_quality(
        engine,
        RepresentationAndQualityInput(
            history_matrix=engine._history_ring.chronological_matrix(),
            history_timestamps=engine._history_ring.chronological_timestamps(),
        ),
    )


def test_scoring_preparation_contract_and_flags() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    _seed(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 3.0}),
            _frame(2.0, {"a": 2.0, "b": 2.0}),
            _frame(3.0, {"a": 3.0, "b": 1.0}),
            _frame(4.0, {"a": 4.0, "b": 0.0}),
            _frame(5.0, {"a": 5.0, "b": -1.0}),
        ],
    )
    rep = _representation(engine)

    result = prepare_scoring_inputs(
        engine,
        ScoringPreparationInput(
            z_baseline=rep.z_baseline,
            z_recent=rep.z_recent,
            valid_mask=rep.valid_mask,
            valid_signal_count=rep.valid_signal_count,
        ),
    )

    assert [f.name for f in fields(ScoringPreparationInput)] == [
        "z_baseline",
        "z_recent",
        "valid_mask",
        "valid_signal_count",
    ]
    assert isinstance(result, ScoringPreparationResult)
    assert result.should_proceed_scoring
    assert result.correlation_ready
    assert result.covariance_ready
    assert result.minimum_sample_ready
    assert result.shape_compatible


def test_scoring_preparation_insufficient_sample_gates_off() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    _seed(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 2.0}),
            _frame(2.0, {"a": 2.0, "b": 2.0}),
            _frame(3.0, {"a": 3.0, "b": 2.0}),
            _frame(4.0, {"a": 4.0, "b": 2.0}),
            _frame(5.0, {"a": 5.0, "b": 2.0}),
        ],
    )
    rep = _representation(engine)
    assert rep.valid_signal_count < 2

    result = prepare_scoring_inputs(
        engine,
        ScoringPreparationInput(
            z_baseline=rep.z_baseline,
            z_recent=rep.z_recent,
            valid_mask=rep.valid_mask,
            valid_signal_count=rep.valid_signal_count,
        ),
    )

    assert not result.should_proceed_scoring
    assert not result.correlation_ready
    assert result.z_baseline_valid is None
    assert result.corr_recent is None


def test_scoring_preparation_rolling_baseline_gating() -> None:
    engine = StructuralEngine(baseline_window=5, recent_window=3)
    _seed(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 3.0, "c": 1.0}),
            _frame(2.0, {"a": 2.0, "b": 4.0, "c": 2.0}),
            _frame(3.0, {"a": 3.0, "b": 5.0, "c": 3.0}),
            _frame(4.0, {"a": 4.0, "b": 6.0, "c": 4.0}),
            _frame(5.0, {"a": 5.0, "b": 7.0, "c": 5.0}),
        ],
    )
    rep = _representation(engine)
    first = prepare_scoring_inputs(
        engine,
        ScoringPreparationInput(rep.z_baseline, rep.z_recent, rep.valid_mask, rep.valid_signal_count),
    )
    assert first.corr_recent is not None

    engine._rolling_baseline_corr = np.ones_like(first.corr_recent, dtype=float) * 0.25
    rolled = prepare_scoring_inputs(
        engine,
        ScoringPreparationInput(rep.z_baseline, rep.z_recent, rep.valid_mask, rep.valid_signal_count),
    )

    assert rolled.baseline_mode == "rolling"
    assert rolled.baseline_corr_used is not None
    assert np.allclose(rolled.baseline_corr_used, engine._rolling_baseline_corr)


def test_scoring_preparation_preserves_inline_parity() -> None:
    engine = StructuralEngine(baseline_window=6, recent_window=4)
    _seed(
        engine,
        [
            _frame(1.0, {"a": 1.0, "b": 2.0, "c": 3.0}),
            _frame(2.0, {"a": 2.0, "b": 1.0, "c": 4.0}),
            _frame(3.0, {"a": 3.0, "b": 2.0, "c": 5.0}),
            _frame(4.0, {"a": 4.0, "b": 3.0, "c": 6.0}),
            _frame(5.0, {"a": 5.0, "b": 4.0, "c": 7.0}),
            _frame(6.0, {"a": 6.0, "b": 5.0, "c": 8.0}),
        ],
    )
    rep = _representation(engine)
    stage_result = prepare_scoring_inputs(
        engine,
        ScoringPreparationInput(rep.z_baseline, rep.z_recent, rep.valid_mask, rep.valid_signal_count),
    )

    z_base_valid = rep.z_baseline[:, rep.valid_mask]
    z_recent_valid = rep.z_recent[:, rep.valid_mask]
    expected_features = FeatureExtractionStage.extract(z_base_valid, z_recent_valid)
    expected_corr_baseline = correlation_matrix(z_base_valid)
    expected_corr_recent = correlation_matrix(z_recent_valid)

    assert stage_result.z_baseline_valid is not None
    assert stage_result.z_recent_valid is not None
    assert stage_result.stage_features is not None
    assert np.allclose(stage_result.z_baseline_valid, z_base_valid, equal_nan=True)
    assert np.allclose(stage_result.z_recent_valid, z_recent_valid, equal_nan=True)
    assert np.allclose(np.asarray(stage_result.corr_baseline), expected_corr_baseline, equal_nan=True)
    assert np.allclose(np.asarray(stage_result.corr_recent), expected_corr_recent, equal_nan=True)
    assert stage_result.stage_features.get("rich_signal_features") == expected_features.get("rich_signal_features")
