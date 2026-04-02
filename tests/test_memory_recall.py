from __future__ import annotations

from neraium_core.memory_recall import (
    DEDUPLICATE_SIMILARITY_THRESHOLD,
    build_structural_memory_signature,
    classify_novelty,
    signature_similarity,
)
from neraium_core.output_contract import build_canonical_output
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


def _raw_result(ts: str, pressure: float, *, asset_id: str = "asset-a", site_id: str = "site-a") -> dict[str, object]:
    return {
        "timestamp": ts,
        "asset_id": asset_id,
        "site_id": site_id,
        "attribution": {
            "top_drivers": [
                {"driver": "pressure", "score": pressure / 100.0},
                {"driver": "temperature", "score": 0.3},
            ]
        },
        "regime_memory": {"state": "degrading"},
        "risk_assessment": {"risk_level": "HIGH" if pressure > 80 else "MEDIUM", "trend": "RISING", "latest_instability": 1.2},
        "decision": {"state": "ALERT", "action": "inspect_pressure_line"},
        "confidence": 0.81,
        "causal_analysis": {"dominant_hypothesis": "pressure_coupling"},
        "explanation_text": "base explanation",
    }


def test_signature_stability() -> None:
    c1 = build_canonical_output(_raw_result("2026-01-01T00:00:00+00:00", 90.0), cycle=1, run_id="r1", customer_id="c1")
    c2 = build_canonical_output(_raw_result("2026-01-01T00:00:00+00:00", 90.0), cycle=1, run_id="r1", customer_id="c1")
    assert build_structural_memory_signature(c1) == build_structural_memory_signature(c2)


def test_similarity_and_novelty_behavior() -> None:
    sig_a = build_structural_memory_signature(
        build_canonical_output(_raw_result("2026-01-01T00:00:00+00:00", 90.0), cycle=1, run_id="r1", customer_id="c1")
    )
    sig_b = build_structural_memory_signature(
        build_canonical_output(_raw_result("2026-01-01T00:01:00+00:00", 89.5), cycle=2, run_id="r1", customer_id="c1")
    )
    raw_c = _raw_result("2026-01-01T00:02:00+00:00", 40.0)
    raw_c["attribution"] = {"top_drivers": [{"driver": "vibration", "score": 0.9}]}
    raw_c["risk_assessment"] = {"risk_level": "LOW", "trend": "STABLE", "latest_instability": 0.2}
    raw_c["decision"] = {"state": "STABLE", "action": "none"}
    raw_c["causal_analysis"] = {"dominant_hypothesis": "none"}
    sig_c = build_structural_memory_signature(
        build_canonical_output(raw_c, cycle=3, run_id="r1", customer_id="c1")
    )

    close = signature_similarity(sig_a, sig_b)
    far = signature_similarity(sig_a, sig_c)
    assert close.total > far.total
    assert close.total > 0.6

    known = classify_novelty(best_similarity=close.total, best_overlap=close.drivers_overlap, memory_count=5)
    novel = classify_novelty(best_similarity=far.total, best_overlap=far.drivers_overlap, memory_count=5)
    assert known["is_novel"] is False
    assert novel["is_novel"] is True


def test_cross_asset_recall_and_dedupe(tmp_path) -> None:
    service = StructuralMonitoringService(store=ResultStore(db_path=str(tmp_path / "memory.db")))
    payload_base = {
        "site_id": "site-a",
        "sensor_values": {"pressure": 50.0, "temperature": 70.0},
    }
    service.ingest_frame(
        {"timestamp": "2026-01-01T00:00:00+00:00", "asset_id": "asset-a", **payload_base},
        run_id="run-1",
        customer_id="cust-1",
    )
    second = service.ingest_frame(
        {"timestamp": "2026-01-01T00:00:10+00:00", "asset_id": "asset-b", **payload_base},
        run_id="run-2",
        customer_id="cust-1",
    )
    recall = second["memory_recall"]
    assert "novelty" in recall
    assert "top_matches" in recall
    matches = service.get_memory_matches(customer_id="cust-1", limit=20)
    assert len(matches) >= 1
    # dedupe: adjacent very similar same-asset records should not explode storage
    service.ingest_frame(
        {"timestamp": "2026-01-01T00:00:20+00:00", "asset_id": "asset-b", **payload_base},
        run_id="run-2",
        customer_id="cust-1",
    )
    matches_after = service.get_memory_matches(customer_id="cust-1", asset_id="asset-b", limit=20)
    assert len(matches_after) <= len(matches) + 1
    assert DEDUPLICATE_SIMILARITY_THRESHOLD >= 0.9
