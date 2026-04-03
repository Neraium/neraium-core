from __future__ import annotations

from datetime import datetime, timedelta, timezone

from neraium_core.trading_signals import IntradaySignalPolicy, build_signal_decision


def _sample_output(score: float = 2.3, confidence: float = 0.8, state: str = "WATCH") -> dict[str, float | str]:
    return {
        "structural_drift_score": score,
        "latest_instability": score,
        "evidence_confidence": confidence,
        "drift_state": state,
    }


def test_low_confidence_is_suppressed() -> None:
    decision = build_signal_decision(
        output=_sample_output(confidence=0.2),
        ticker="AAPL",
        timestamp=datetime(2026, 4, 2, tzinfo=timezone.utc),
        policy=IntradaySignalPolicy(min_confidence=0.5, warmup_bars=1),
        previous=None,
        bars_seen=3,
    )
    assert decision["emit"] is False
    assert decision["reason"] == "confidence_below_gate"


def test_duplicate_signal_within_cooldown_is_suppressed() -> None:
    now = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    decision = build_signal_decision(
        output=_sample_output(),
        ticker="AAPL",
        timestamp=now,
        policy=IntradaySignalPolicy(cooldown_seconds=60, warmup_bars=1),
        previous={
            "timestamp": now - timedelta(seconds=20),
            "trading_signal": "REDUCE",
            "state": "WATCH",
            "structural_drift_score": 2.3,
        },
        bars_seen=5,
    )
    assert decision["emit"] is False
    assert decision["reason"] == "duplicate_within_cooldown"


def test_state_transition_is_marked() -> None:
    now = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)
    decision = build_signal_decision(
        output=_sample_output(state="ALERT", score=3.2),
        ticker="AAPL",
        timestamp=now,
        policy=IntradaySignalPolicy(warmup_bars=1),
        previous={
            "timestamp": now - timedelta(seconds=120),
            "trading_signal": "REDUCE",
            "state": "WATCH",
            "structural_drift_score": 2.1,
        },
        bars_seen=8,
    )
    assert decision["transition"] is True
    assert decision["emit"] is True
