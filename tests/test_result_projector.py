from __future__ import annotations

import pytest

from neraium_core.output_contract import is_canonical_output
from neraium_core.pilot_config import load_pilot_config
from neraium_core.result_projector import CanonicalOutputProjector


def _sample_engine_result() -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "site-a",
        "asset_id": "asset-a",
        "state": "STABLE",
        "structural_drift_score": 0.25,
        "transition_pressure": 0.86,
        "transition_state": "EMERGING_TRANSITION",
        "latest_instability": 0.42,
        "relational_stability_score": 0.73,
        "sensor_relationships": [{"a": 1}, {"b": 2}],
        "experimental_analytics": {"forecasting": {"trend": 0.1}},
    }


def _payload() -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "site-a",
        "asset_id": "asset-a",
        "sensor_values": {"pressure": 41.5, "temperature": 72.1},
    }


def test_projector_canonical_output_shape() -> None:
    projector = CanonicalOutputProjector(load_pilot_config())
    decorated = projector.project_for_output(_sample_engine_result())

    canonical = projector.project_canonical_output(
        decorated,
        cycle=1,
        run_id="run-a",
        customer_id="cust-a",
        previous=None,
        memory_recall=None,
    )

    assert is_canonical_output(canonical)
    assert canonical["session"]["run_id"] == "run-a"
    assert canonical["session"]["customer_id"] == "cust-a"


def test_projector_handles_missing_optional_fields() -> None:
    projector = CanonicalOutputProjector(load_pilot_config())
    decorated = projector.project_for_output({"timestamp": "2026-01-01T00:00:00+00:00"})

    assert decorated["trend"] == "UNKNOWN"
    assert decorated["confidence"] == 0.0
    assert decorated["risk_level"] == "LOW"
    assert decorated["structural_analysis_available"] is False


def test_service_decorate_result_facade_compatibility(tmp_path) -> None:
    pytest.importorskip("numpy")
    from neraium_core.alignment import StructuralEngine
    from neraium_core.service import StructuralMonitoringService
    from neraium_core.store import ResultStore

    service = StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3),
        store=ResultStore(db_path=str(tmp_path / "projector_facade.db")),
    )
    raw = _sample_engine_result()

    via_service = service._decorate_result(raw)  # pylint: disable=protected-access
    via_projector = service._projector.project_for_output(raw)  # pylint: disable=protected-access

    assert via_service == via_projector


def test_ingest_payload_and_batch_projection_consistency(tmp_path) -> None:
    pytest.importorskip("numpy")
    from neraium_core.alignment import StructuralEngine
    from neraium_core.service import StructuralMonitoringService
    from neraium_core.store import ResultStore

    payload_service = StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3),
        store=ResultStore(db_path=str(tmp_path / "payload.db")),
    )
    batch_service = StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3),
        store=ResultStore(db_path=str(tmp_path / "batch.db")),
    )

    payload_result = payload_service.ingest_payload(dict(_payload()), run_id="run-a", customer_id="cust-a")
    batch_result = batch_service.ingest_batch([dict(_payload())], run_id="run-a", customer_id="cust-a")[0]

    for key in ("risk_level", "action_state", "operator_message", "trend", "confidence"):
        assert payload_result[key] == batch_result[key]
    assert payload_result["interpretation"] == batch_result["interpretation"]
