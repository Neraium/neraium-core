from __future__ import annotations

from typing import Any, Mapping


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


def build_memory_context_text(memory_recall: Mapping[str, Any] | None) -> str | None:
    if not isinstance(memory_recall, Mapping):
        return None

    novelty = memory_recall.get("novelty") if isinstance(memory_recall.get("novelty"), Mapping) else {}
    nearest = memory_recall.get("nearest_match") if isinstance(memory_recall.get("nearest_match"), Mapping) else {}

    if bool(nearest.get("found", False)):
        similarity = float(nearest.get("similarity", 0.0) or 0.0)
        asset_id = nearest.get("asset_id")
        run_id = nearest.get("run_id")
        cycle = nearest.get("cycle")
        target = "prior stored structure"
        if asset_id:
            target = f"asset {asset_id}"
        details = []
        if run_id:
            details.append(f"run {run_id}")
        if cycle is not None:
            details.append(f"cycle {cycle}")
        extra = f" ({', '.join(details)})" if details else ""
        return f"Current structure resembles a prior pattern from {target}{extra} (similarity {similarity:.2f})."

    if bool(novelty.get("is_novel", False)):
        reason = str(novelty.get("reason", "limited historical overlap")).replace("_", " ")
        return f"Current structure appears novel relative to stored history ({reason})."
    return None


def build_explanation_text(
    *,
    current_decision: str,
    attribution: dict[str, Any] | None,
    risk: str | float | int | None,
    confidence: str | float | int | None,
    recommended_action: str | None = None,
    memory_recall: Mapping[str, Any] | None = None,
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

    memory_line = build_memory_context_text(memory_recall)
    if memory_line:
        sentences.append(memory_line)

    action = str(recommended_action or "").strip()
    if action:
        sentences.append(f"Recommended action: {action}.")

    return " ".join(sentences[:5])
