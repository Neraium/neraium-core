from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class IntradaySignalPolicy:
    min_confidence: float = 0.3
    warmup_bars: int = 5
    cooldown_seconds: int = 120


def build_signal_decision(
    *,
    output: dict[str, Any],
    ticker: str,
    timestamp: datetime,
    policy: IntradaySignalPolicy,
    previous: dict[str, Any] | None,
    bars_seen: int,
) -> dict[str, Any]:
    confidence = float(output.get("evidence_confidence", 0.0))
    state = str(output.get("drift_state", "NORMAL")).upper()
    score = float(output.get("structural_drift_score", output.get("latest_instability", 0.0)))

    signal = "HOLD"
    if state in {"WATCH", "ALERT"} or score >= 2.0:
        signal = "REDUCE"

    if bars_seen < policy.warmup_bars:
        return {
            "ticker": ticker,
            "timestamp": timestamp,
            "emit": False,
            "reason": "warmup",
            "transition": False,
            "state": state,
            "trading_signal": "WARMUP",
            "confidence": confidence,
            "structural_drift_score": score,
            "latest_instability": float(output.get("latest_instability", score)),
        }

    if confidence < policy.min_confidence:
        emit = False
        reason = "confidence_below_gate"
    elif previous and previous.get("trading_signal") == signal:
        prev_ts = previous.get("timestamp")
        elapsed = (timestamp - prev_ts).total_seconds() if isinstance(prev_ts, datetime) else float("inf")
        if elapsed < policy.cooldown_seconds:
            emit = False
            reason = "duplicate_within_cooldown"
        else:
            emit = True
            reason = "signal_eligible"
    else:
        emit = True
        reason = "signal_eligible"

    transition = bool(previous and previous.get("state") != state)
    return {
        "ticker": ticker,
        "timestamp": timestamp,
        "emit": emit,
        "reason": reason,
        "transition": transition,
        "state": state,
        "trading_signal": signal,
        "confidence": confidence,
        "structural_drift_score": score,
        "latest_instability": float(output.get("latest_instability", score)),
    }
