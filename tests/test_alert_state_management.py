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


def _next(
    previous: dict[str, object] | None,
    cycle: int,
    *,
    risk_level: str,
    trend: str = "STABLE",
    alert_control: dict[str, object] | None = None,
    alert_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    raw = _raw_result(risk_level=risk_level, trend=trend)
    if alert_control:
        raw["alert_control"] = alert_control
    if alert_policy:
        raw["alert_policy"] = alert_policy
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
    assert s5["alert_status"]["state"] == "ACTIVE_UNACKNOWLEDGED"


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
    assert clean3["alert_status"]["alert_summary"] == "Resolved after sustained recovery"


def test_manual_resolution_path() -> None:
    a1 = _next(None, 1, risk_level="HIGH")
    a2 = _next(a1, 2, risk_level="HIGH")
    active = _next(a2, 3, risk_level="HIGH")

    resolved = _next(active, 4, risk_level="HIGH", alert_control={"resolve": True, "resolved_by": "operator-b"})
    assert resolved["alert_status"]["alert_state"] == "RESOLVED"
    assert resolved["alert_status"]["resolved_reason"] == "manual_resolution"
    assert resolved["alert_status"]["resolved_by"] == "operator-b"


def test_pending_activation_exactly_at_three_hits() -> None:
    p1 = _next(None, 1, risk_level="HIGH")
    p2 = _next(p1, 2, risk_level="HIGH")
    assert p1["alert_status"]["alert_state"] == "PENDING_ALERT"
    assert p2["alert_status"]["alert_state"] == "PENDING_ALERT"
    assert p2["alert_status"]["consecutive_hit_count"] == 2
    p3 = _next(p2, 3, risk_level="HIGH")
    assert p3["alert_status"]["alert_state"] == "ACTIVE_UNACKNOWLEDGED"
    assert p3["alert_status"]["consecutive_hit_count"] == 3


def test_acknowledged_active_alert_remains_active_until_resolved() -> None:
    a1 = _next(None, 1, risk_level="HIGH")
    a2 = _next(a1, 2, risk_level="HIGH")
    active = _next(a2, 3, risk_level="HIGH")
    acknowledged = _next(active, 4, risk_level="HIGH", alert_control={"acknowledge": True, "acknowledged_by": "operator-a"})
    s5 = _next(acknowledged, 5, risk_level="HIGH")
    s6 = _next(s5, 6, risk_level="HIGH")
    assert s5["alert_status"]["alert_state"] == "ACTIVE_ACKNOWLEDGED"
    assert s6["alert_status"]["alert_state"] == "ACTIVE_ACKNOWLEDGED"
    assert s6["alert_status"]["acknowledged"] is True


def test_resolved_state_is_stable_without_new_candidate() -> None:
    a1 = _next(None, 1, risk_level="HIGH")
    a2 = _next(a1, 2, risk_level="HIGH")
    active = _next(a2, 3, risk_level="HIGH")
    clean1 = _next(active, 4, risk_level="LOW")
    clean2 = _next(clean1, 5, risk_level="LOW")
    resolved = _next(clean2, 6, risk_level="LOW")
    still_resolved = _next(resolved, 7, risk_level="LOW")
    assert resolved["alert_status"]["alert_state"] == "RESOLVED"
    assert still_resolved["alert_status"]["alert_state"] == "RESOLVED"


def test_renotify_cadence_does_not_spam_every_cycle() -> None:
    a1 = _next(None, 1, risk_level="HIGH")
    a2 = _next(a1, 2, risk_level="HIGH")
    active = _next(a2, 3, risk_level="HIGH")
    c4 = _next(active, 4, risk_level="HIGH")
    c5 = _next(c4, 5, risk_level="HIGH")
    c6 = _next(c5, 6, risk_level="HIGH")
    c7 = _next(c6, 7, risk_level="HIGH")
    assert c4["alert_status"]["renotify_due"] is False
    assert c5["alert_status"]["renotify_due"] is False
    assert c6["alert_status"]["renotify_due"] is True
    assert c7["alert_status"]["renotify_due"] is False


def test_escalation_occurs_after_prolonged_unacknowledged_duration() -> None:
    s = _next(None, 1, risk_level="HIGH")
    s = _next(s, 2, risk_level="HIGH")
    s = _next(s, 3, risk_level="HIGH")
    for cycle in range(4, 10):
        s = _next(s, cycle, risk_level="HIGH")
    assert s["alert_status"]["alert_state"] == "ESCALATED"
    assert s["alert_status"]["escalation_reason"] in {
        "prolonged_unacknowledged_alert",
        "repeated_renotify_without_acknowledgement",
        "worsening_risk_severity",
    }


def test_custom_alert_policy_can_change_trigger_and_resolve_thresholds() -> None:
    policy = {
        "trigger_hit_threshold": 4,
        "resolve_clean_window_threshold": 2,
        "renotify_interval_cycles": 2,
        "escalation_unacknowledged_cycles": 8,
    }
    s1 = _next(None, 1, risk_level="HIGH", alert_policy=policy)
    s2 = _next(s1, 2, risk_level="HIGH", alert_policy=policy)
    s3 = _next(s2, 3, risk_level="HIGH", alert_policy=policy)
    assert s3["alert_status"]["alert_state"] == "PENDING_ALERT"
    assert s3["alert_status"]["consecutive_hit_count"] == 3

    active = _next(s3, 4, risk_level="HIGH", alert_policy=policy)
    assert active["alert_status"]["alert_state"] == "ACTIVE_UNACKNOWLEDGED"
    assert active["alert_status"]["policy"]["trigger_hit_threshold"] == 4

    clean1 = _next(active, 5, risk_level="LOW", alert_policy=policy)
    resolved = _next(clean1, 6, risk_level="LOW", alert_policy=policy)
    assert resolved["alert_status"]["alert_state"] == "RESOLVED"
    assert resolved["alert_status"]["resolution_hit_count"] == 2
