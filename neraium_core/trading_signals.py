from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


def map_structural_output_to_signal(output: dict[str, Any] | None) -> str:
    """Map engine output into a conservative action label."""
    if not output:
        return "HOLD"

    score = float(output.get("structural_drift_score", output.get("latest_instability", 0.0)) or 0.0)
    state = str(output.get("drift_state", output.get("classification", ""))).upper()
    health = output.get("system_health")
    try:
        health_value = float(health) if health is not None else None
    except (TypeError, ValueError):
        health_value = None

    if state == "ALERT" or score >= 3.0 or (health_value is not None and health_value < 25.0):
        return "EXIT"
    if state == "WATCH" or score >= 2.0:
        return "REDUCE"

    confidence = output.get("evidence_confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence_value = 0.0

    if score <= 0.6 and confidence_value >= 0.5:
        return "BUY"
    if score >= 1.2:
        return "SELL"
    return "HOLD"


@dataclass(frozen=True)
class IntradaySignalPolicy:
    min_confidence: float = 0.45
    cooldown_seconds: float = 45.0
    emit_on_state_change_only: bool = False
    drift_enter_threshold: float = 2.0
    drift_exit_threshold: float = 1.7
    warmup_bars: int = 5


def build_signal_decision(
    *,
    output: dict[str, Any] | None,
    ticker: str,
    timestamp: datetime,
    policy: IntradaySignalPolicy,
    previous: dict[str, Any] | None,
    bars_seen: int,
) -> dict[str, Any]:
    result = output or {}
    score = float(result.get("structural_drift_score", result.get("latest_instability", 0.0)) or 0.0)
    confidence = float(result.get("evidence_confidence", result.get("confidence", 0.0)) or 0.0)
    state = str(result.get("drift_state", result.get("classification", "UNKNOWN"))).upper()
    raw_signal = map_structural_output_to_signal(result)

    reason = "signal_eligible"
    emit = True
    transition = False
    cooldown_remaining_seconds = 0.0
    signal = raw_signal

    if bars_seen < policy.warmup_bars:
        emit = False
        signal = "WARMUP"
        reason = "warmup_incomplete"
    elif confidence < policy.min_confidence:
        emit = False
        signal = "HOLD"
        reason = "confidence_below_gate"
    else:
        prev_signal = str((previous or {}).get("trading_signal", ""))
        prev_state = str((previous or {}).get("state", ""))
        prev_ts = (previous or {}).get("timestamp")
        prev_score = float((previous or {}).get("structural_drift_score", 0.0) or 0.0)

        if isinstance(prev_ts, datetime):
            elapsed = (timestamp - prev_ts).total_seconds()
            if elapsed < policy.cooldown_seconds and signal == prev_signal and state == prev_state:
                emit = False
                cooldown_remaining_seconds = max(0.0, policy.cooldown_seconds - elapsed)
                reason = "duplicate_within_cooldown"

        if state != prev_state and prev_state:
            transition = True
            reason = "state_transition"

        if policy.emit_on_state_change_only and not transition:
            emit = False
            reason = "state_change_only_mode"

        if prev_state == "WATCH" and state == "NORMAL":
            if score >= policy.drift_exit_threshold or prev_score >= policy.drift_enter_threshold:
                emit = False
                reason = "hysteresis_hold"

    return {
        "ticker": ticker,
        "timestamp": timestamp,
        "state": state,
        "trading_signal": signal,
        "confidence": round(max(0.0, min(confidence, 1.0)), 4),
        "structural_drift_score": round(score, 6),
        "latest_instability": float(result.get("latest_instability", score) or score),
        "emit": emit,
        "transition": transition,
        "reason": reason,
        "cooldown_remaining_seconds": round(cooldown_remaining_seconds, 3),
        "policy": {
            "min_confidence": policy.min_confidence,
            "cooldown_seconds": policy.cooldown_seconds,
            "emit_on_state_change_only": policy.emit_on_state_change_only,
            "warmup_bars": policy.warmup_bars,
        },
    }


__all__ = ["map_structural_output_to_signal", "IntradaySignalPolicy", "build_signal_decision"]
