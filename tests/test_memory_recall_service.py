from __future__ import annotations

from neraium_core.memory_recall_service import MemoryRecallService
from neraium_core.pilot_config import load_pilot_config
from neraium_core.result_projector import CanonicalOutputProjector
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _service_parts(tmp_path) -> tuple[ResultStore, MemoryRecallService]:
    store = ResultStore(db_path=str(tmp_path / "memory_recall_component.db"))
    projector = CanonicalOutputProjector(load_pilot_config())
    return store, MemoryRecallService(store, projector)


def _result(ts: str, *, asset_id: str = "asset-1", site_id: str = "site-1") -> dict[str, object]:
    return {
        "timestamp": ts,
        "asset_id": asset_id,
        "site_id": site_id,
        "sensor_relationships": ["pressure", "temperature"],
        "experimental_analytics": {"relational_metrics_skipped": False},
        "attribution": {"top_drivers": [{"driver": "pressure", "score": 0.9}]},
        "risk_assessment": {"risk_level": "MEDIUM", "trend": "RISING", "latest_instability": 1.1},
        "decision": {"state": "WATCH", "action": "inspect_valve"},
        "state": "WATCH",
    }


def test_recall_payload_shape_when_history_missing(tmp_path) -> None:
    _, recall_service = _service_parts(tmp_path)
    payload = recall_service.build_memory_recall(
        _result("2026-04-03T00:00:00+00:00"),
        cycle=1,
        run_id="run-a",
        customer_id="cust-a",
    )
    assert set(payload.keys()) == {"status", "signature", "novelty", "nearest_match", "top_matches", "pattern_family"}
    assert payload["status"]["memory_records_considered"] == 0
    assert payload["nearest_match"]["found"] is False
    assert payload["top_matches"] == []


def test_recall_with_partial_result_is_still_built(tmp_path) -> None:
    _, recall_service = _service_parts(tmp_path)
    partial = {
        "timestamp": "2026-04-03T00:00:00+00:00",
        "site_id": "site-1",
        "asset_id": "asset-1",
    }
    payload = recall_service.build_memory_recall(partial, cycle=1, run_id="run-a", customer_id="cust-a")
    assert isinstance(payload.get("signature"), dict)
    assert "risk_level" in payload["signature"]


def test_service_facade_memory_recall_compatibility(tmp_path) -> None:
    service = StructuralMonitoringService(store=ResultStore(db_path=str(tmp_path / "service_facade.db")))
    result = _result("2026-04-03T00:00:00+00:00")
    via_facade = service._build_memory_recall(result, cycle=1, run_id="run-a", customer_id="cust-a")
    via_component = service._memory_recall.build_memory_recall(result, cycle=1, run_id="run-a", customer_id="cust-a")
    assert via_facade == via_component


def test_persist_structural_memory_dedupe_policy(tmp_path) -> None:
    service = StructuralMonitoringService(store=ResultStore(db_path=str(tmp_path / "dedupe.db")))
    canonical = service._projector.project_canonical_output(
        _result("2026-04-03T00:00:00+00:00"),
        cycle=1,
        run_id="run-a",
        customer_id="cust-a",
        previous=None,
        memory_recall=None,
    )
    memory_recall = service._memory_recall.build_memory_recall(
        _result("2026-04-03T00:00:00+00:00"),
        cycle=1,
        run_id="run-a",
        customer_id="cust-a",
    )
    memory_recall["nearest_match"] = {"similarity": 1.0, "scope": "same_asset", "memory_id": 10}
    persisted = service._persist_structural_memory(canonical, memory_recall, customer_id="cust-a")
    assert persisted is None
