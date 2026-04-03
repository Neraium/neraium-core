from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def compact_intraday_output(
    *,
    ticker: str,
    timestamp: datetime,
    decision: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    state = str(decision.get("state", result.get("drift_state", result.get("classification", "UNKNOWN")))).upper()
    reason = str(decision.get("reason") or result.get("explanation_text") or "No explanation available.")
    return {
        "timestamp": timestamp.isoformat(),
        "ticker": ticker,
        "state": state,
        "trading_signal": str(decision.get("trading_signal", "HOLD")),
        "confidence": float(decision.get("confidence", result.get("evidence_confidence", 0.0)) or 0.0),
        "structural_drift_score": float(decision.get("structural_drift_score", result.get("structural_drift_score", 0.0)) or 0.0),
        "latest_instability": float(decision.get("latest_instability", result.get("latest_instability", 0.0)) or 0.0),
        "reason": reason,
        "transition": bool(decision.get("transition", False)),
        "cooldown_remaining_seconds": float(decision.get("cooldown_remaining_seconds", 0.0) or 0.0),
        "emitted": bool(decision.get("emit", True)),
    }
