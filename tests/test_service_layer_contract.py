from __future__ import annotations

from neraium_core.alignment import StructuralEngine
from neraium_core.output_contract import (
    CANONICAL_SCHEMA_VERSION,
    REQUIRED_FIELDS,
    derive_product_events,
    is_canonical_output,
    build_decision_contract_v2,
)
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _service(tmp_path) -> StructuralMonitoringService:
    return StructuralMonitoringService(
        engine=StructuralEngine(baseline_window=5, recent_window=3),
        store=ResultStore(db_path=str(tmp_path / "service_layer.db")),
    )


def _frame(i: int, pressure: float) -> dict[str, object]:
    return {
        "timestamp": f"2026-01-01T00:00:{i:02d}+00:00",
        "site_id": "plant-1",
        "asset_id": "pump-9",
        "sensor_values": {
            "pressure": pressure,
            "temperature": 80.0 + 0.5 * i,
        },
    }


def test_ingest_frame_returns_canonical_contract_and_persists_history(tmp_path) -> None:
    service = _service(tmp_path)

    out = service.ingest_frame(_frame(0, 50.0), run_id="run-prod", customer_id="customer-a")

    assert is_canonical_output(out)
    assert out["schema_version"] == CANONICAL_SCHEMA_VERSION
    assert REQUIRED_FIELDS.issubset(set(out.keys()))
    assert "memory_recall" in out
    assert "novelty" in out["memory_recall"]

    latest = service.get_current_state(run_id="run-prod", customer_id="customer-a")
    assert latest is not None
    assert latest["history_id"] == out["history_id"]


def test_recent_history_and_latest_helpers(tmp_path) -> None:
    service = _service(tmp_path)

    service.ingest_batch(
        [_frame(0, 50.0), _frame(1, 53.0), _frame(2, 56.0)],
        run_id="run-prod",
        customer_id="customer-a",
    )

    history = service.get_recent_history(limit=2, run_id="run-prod", customer_id="customer-a")
    assert len(history) == 2
    assert history[0]["cycle"] > history[1]["cycle"]

    recommendation = service.get_latest_recommendation(run_id="run-prod", customer_id="customer-a")
    explanation = service.get_latest_explanation(run_id="run-prod", customer_id="customer-a")

    assert isinstance(recommendation, dict)
    assert recommendation["status"]["advisory"] is True
    assert "recommended_action" in recommendation
    assert isinstance(explanation, str)


def test_event_generation_includes_stable_product_flags(tmp_path) -> None:
    service = _service(tmp_path)

    first = service.ingest_frame(_frame(0, 50.0), run_id="run-prod", customer_id="customer-a")
    second = service.ingest_frame(_frame(1, 150.0), run_id="run-prod", customer_id="customer-a")

    assert "recommendation_available" in first["events"]
    assert "recommendation_available" in second["events"]
    synthetic_prev = {
        "risk_assessment": {"risk_level": "LOW", "trend": "STABLE", "latest_instability": 0.2},
        "operational_recommendation": {"status": {"available": True, "advisory": True, "reason": "recommendation_available"}, "recommended_action": "none"},
    }
    synthetic_curr = {
        "risk_assessment": {"risk_level": "HIGH", "trend": "RISING", "latest_instability": 1.5},
        "operational_recommendation": {"status": {"available": True, "advisory": True, "reason": "recommendation_available"}, "recommended_action": "inspect_cooling_loop"},
    }
    events = derive_product_events(synthetic_curr, previous=synthetic_prev)
    assert "risk_escalated" in events
    assert "inspection_recommended" in events
    assert "deterioration_detected" in events


def test_contract_stability_required_and_optional_keys(tmp_path) -> None:
    service = _service(tmp_path)
    out = service.ingest_frame(_frame(0, 50.0), run_id="run-prod", customer_id="customer-a")

    required = REQUIRED_FIELDS
    optional = {"aliases", "session", "history_id", "persisted_at", "customer_id", "run_id", "memory_recall", "operational_recommendation", "recommendation_available", "alert_status"}
    unknown = set(out.keys()) - required - optional
    assert not unknown


def test_decision_contract_v2_is_auditable_and_explicit(tmp_path) -> None:
    service = _service(tmp_path)
    out = service.ingest_frame(_frame(0, 50.0), run_id="run-prod", customer_id="customer-a")

    contract = build_decision_contract_v2(out)
    assert contract is not None
    assert contract["contract_version"] == "decision-contract.v2"
    assert "structural_state" in contract
    assert "confidence_quality" in contract
    assert "evidence_explanation" in contract
    assert "operator_action" in contract
    assert "policy" in contract
    assert "data_quality" in contract
    assert isinstance(contract["confidence_quality"]["unknowns"], list)
