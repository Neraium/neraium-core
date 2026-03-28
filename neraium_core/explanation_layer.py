from __future__ import annotations

from typing import Any


def _to_confidence_label(confidence: str | float | int | None) -> str:
    if isinstance(confidence, str):
        value = confidence.strip().lower()
        if value in {"high", "medium", "low"}:
            return value
    try:
        score = float(confidence)
    except (TypeError, ValueError):
        return "unknown"

    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def build_explanation_text(
    *,
    current_decision: str,
    attribution: dict[str, Any] | None,
    risk: str | float | int | None,
    confidence: str | float | int | None,
    recommended_action: str | None = None,
) -> str:
    """Create a concise, signal-grounded explanation string for pipeline outputs."""
    decision = str(current_decision or "NOMINAL_STRUCTURE").strip() or "NOMINAL_STRUCTURE"

    top_driver = "no dominant driver"
    if isinstance(attribution, dict):
        top_drivers = attribution.get("top_drivers")
        if isinstance(top_drivers, list) and top_drivers:
            top_driver = str(top_drivers[0])

    risk_text = str(risk).strip() if risk is not None else "UNKNOWN"
    confidence_text = _to_confidence_label(confidence)

    sentences = [
        f"Current state is {decision} with {risk_text.lower()} risk.",
        f"Primary driver is {top_driver} based on attribution signals.",
        f"Confidence is {confidence_text}.",
    ]

    action = str(recommended_action or "").strip()
    if action:
        sentences.append(f"Recommended action: {action}.")

    return " ".join(sentences[:4])
