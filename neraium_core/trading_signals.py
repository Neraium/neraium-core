from __future__ import annotations

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


__all__ = ["map_structural_output_to_signal"]
