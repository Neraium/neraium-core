"""Tests for Day 11 alerting."""

from __future__ import annotations

from pathlib import Path

from neraium.alerts import (
    build_alert_record,
    build_operator_inbox,
    detect_alert_conditions,
    load_recent_alerts,
    log_alert_record,
    should_suppress_alert,
)


def test_alert_detection_output() -> None:
    cur = {
        "timestamp": "2024-01-02T00:00:00",
        "market_regime_label": "market_risk_off",
        "market_confidence_score": 0.5,
        "market_warning_level": "high",
        "path_adjusted_market_action": "defensive",
        "market_explanation": "x",
        "scenario_path_label": "unclassified",
    }
    prior = {
        "timestamp": "2024-01-01T00:00:00",
        "market_regime_label": "market_stable",
        "market_confidence_score": 0.7,
        "market_warning_level": "none",
        "path_adjusted_market_action": "risk_on",
        "market_explanation": "y",
        "scenario_path_label": "unclassified",
    }
    out = detect_alert_conditions(cur, prior)
    assert "alert_triggered" in out
    assert "alert_type" in out
    assert out["alert_triggered"] is True


def test_alert_record_created() -> None:
    snap = {
        "timestamp": "2024-01-01T00:00:00",
        "market_regime_label": "market_mixed",
        "market_confidence_score": 0.6,
        "market_warning_level": "low",
        "path_adjusted_market_action": "neutral",
    }
    info = {
        "alert_triggered": True,
        "alert_type": "regime_change",
        "alert_severity": "medium",
        "alert_message": "test",
        "reason_codes": ["regime_change"],
    }
    rec = build_alert_record(snap, info)
    assert rec.get("alert_id")
    assert rec.get("alert_message") == "test"


def test_alert_log_created_and_load(tmp_path: Path) -> None:
    p = tmp_path / "alerts.jsonl"
    rec = {
        "alert_id": "a1",
        "timestamp": "2024-01-01T00:00:00",
        "alert_type": "regime_change",
        "alert_severity": "low",
        "market_regime_label": "x",
        "market_warning_level": "none",
        "path_adjusted_market_action": "wait",
        "market_confidence_score": 0.5,
        "alert_message": "m",
        "reason_codes": [],
    }
    log_alert_record(rec, p)
    assert p.is_file()
    loaded = load_recent_alerts(p, limit=20)
    assert len(loaded) == 1
    assert loaded[0]["alert_id"] == "a1"


def test_alert_suppression_logic() -> None:
    cur = {
        "alert_id": "b",
        "timestamp": "2024-01-02T00:00:00",
        "alert_type": "regime_change",
        "alert_severity": "medium",
        "market_regime_label": "market_mixed",
        "market_warning_level": "none",
        "path_adjusted_market_action": "wait",
        "market_confidence_score": 0.5,
        "alert_message": "dup",
        "reason_codes": [],
    }
    recent = [dict(cur)]
    assert should_suppress_alert(cur, recent) is True


def test_operator_inbox_sorted_and_columns() -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-02"]),
            "market_regime_label": ["a", "b", "c"],
            "market_warning_level": ["none"] * 3,
            "path_adjusted_market_action": ["neutral"] * 3,
            "market_confidence_score": [0.5, 0.6, 0.55],
            "path_aware_market_explanation": ["e"] * 3,
        }
    )
    inbox = build_operator_inbox(df, alert_log_path=Path(__file__).parent / "nonexistent_alerts.jsonl")
    assert not inbox.empty
    assert inbox["timestamp"].is_monotonic_decreasing
    for c in ("alert_type", "alert_severity", "explanation"):
        assert c in inbox.columns
