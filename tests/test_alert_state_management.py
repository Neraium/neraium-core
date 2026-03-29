from __future__ import annotations

from neraium_core.output_contract import build_canonical_output


def _raw_result(*, risk_level: str, trend: str = "STABLE", confidence: float = 0.7, recommendation_available: bool = True, transition_pressure: float = 0.0) -> dict[str, object]:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "site_id": "site-a",
        "asset_id": "asset-a",
        "risk_assessment": {
            "risk_level": risk_level,
            "trend": trend,
            "latest_instability": 1.0 if risk_level == "HIGH" else 0.2,
        },
        "confidence": confidence,
        "transition_pressure": transition_pressure,
        "operational_recommendation": {
            "status": {"available": recommendation_available, "advisory": True, "reason": "recommendation_available"},
            "recommended_action": "inspect_cooling_loop" if recommendation_available else None,
            "rationale": "deterministic test",
        },
        "attribution": {"top_drivers": []},
        "regime_memory": {},
        "causal_analysis": {},
        "explanation_text": "test",
    }


def _next(previous: dict[str, object] | None, cycle: int, *, risk_level: str, trend: str = "STABLE", alert_control: dict[str, object] | None = None) -> dict[str, object]:
    raw = _raw_result(risk_level=risk_level, trend=trend)
    if alert_control:
        raw["alert_control"] = alert_control
    raw["timestamp"] = f"2026-01-01T00:00:{cycle:02d}+00:00"
    return build_canonical_output(raw, cycle=cycle, run_id="run-a", customer_id="customer-a", previous=previous)


def test_three_consecutive_hits_required_and_counter_resets() -> None:
    s1 = _next(None, 1, risk_level="HIGH")
    assert s1["alert_status"]["alert_state"] == "PENDING_ALERT"
    assert s1["alert_status"]["consecutive_hit_count"] == 1

    s2 = _next(s1, 2, risk_level="HIGH")
    assert s2["alert_status"]["alert_state"] == "PENDING_ALERT"
    assert s2["alert_status"]["consecutive_hit_count"] == 2

    s_break = _next(s2, 3, risk_level="LOW")
    assert s_break["alert_status"]["alert_state"] == "CLEAR"
    assert s_break["alert_status"]["consecutive_hit_count"] == 0

    s3 = _next(s_break, 4, risk_level="HIGH")
    s4 = _next(s3, 5, risk_level="HIGH")
    s5 = _next(s4, 6, risk_level="HIGH")
    assert s5["alert_status"]["alert_state"] == "ACTIVE_UNACKNOWLEDGED"
    assert s5["alert_status"]["consecutive_hit_count"] == 3


def test_active_persists_acknowledges_and_resolves_after_three_clean_windows() -> None:
    a1 = _next(None, 1, risk_level="HIGH")
    a2 = _next(a1, 2, risk_level="HIGH")
    active = _next(a2, 3, risk_level="HIGH")
    assert active["alert_status"]["alert_state"] == "ACTIVE_UNACKNOWLEDGED"

    acknowledged = _next(active, 4, risk_level="HIGH", alert_control={"acknowledge": True, "acknowledged_by": "operator-a"})
    assert acknowledged["alert_status"]["alert_state"] == "ACTIVE_ACKNOWLEDGED"
    assert acknowledged["alert_status"]["acknowledged"] is True
    assert acknowledged["alert_status"]["acknowledged_by"] == "operator-a"

    clean1 = _next(acknowledged, 5, risk_level="LOW")
    clean2 = _next(clean1, 6, risk_level="LOW")
    clean3 = _next(clean2, 7, risk_level="LOW")
    assert clean3["alert_status"]["alert_state"] == "RESOLVED"
    assert clean3["alert_status"]["resolution_hit_count"] == 3
    assert clean3["alert_status"]["resolved_reason"] == "auto_resolved_after_3_clean_windows"


def test_manual_resolution_path() -> None:
    a1 = _next(None, 1, risk_level="HIGH")
    a2 = _next(a1, 2, risk_level="HIGH")
    active = _next(a2, 3, risk_level="HIGH")

    resolved = _next(active, 4, risk_level="HIGH", alert_control={"resolve": True, "resolved_by": "operator-b"})
    assert resolved["alert_status"]["alert_state"] == "RESOLVED"
    assert resolved["alert_status"]["resolved_reason"] == "manual_resolution"
    assert resolved["alert_status"]["resolved_by"] == "operator-b"
