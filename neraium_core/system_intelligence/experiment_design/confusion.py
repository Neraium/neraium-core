from __future__ import annotations

from typing import Any


def detect_structural_confusion(
    *,
    hypotheses: list[dict[str, Any]],
    trajectory: dict[str, Any],
    reliability: dict[str, Any],
) -> dict[str, Any]:
    confusion_types: list[str] = []
    actions: list[str] = []

    if len(hypotheses) >= 2:
        gap = abs(float(hypotheses[0].get("likelihood", 0.0)) - float(hypotheses[1].get("likelihood", 0.0)))
        if gap <= 0.1:
            confusion_types.append("hypothesis_tie")
            actions.append("Run highest discriminative low-risk experiment.")

    contradiction = max((float(h.get("contradiction_rate", 0.0)) for h in hypotheses), default=0.0)
    if contradiction >= 0.45:
        confusion_types.append("high_contradiction")
        actions.append("Audit assumptions and sensor quality before acting.")

    novelty = float(trajectory.get("novelty_score", 0.0))
    support = float(trajectory.get("support_count", 0.0))
    if novelty >= 0.75 and support <= 2:
        confusion_types.append("novel_low_support")
        actions.append("Increase measurement density; treat recommendations as provisional.")

    calibrated = float(((reliability.get("intervention_recommendation") or {}).get("recommendation_calibrated_confidence", 0.0)))
    if calibrated <= 0.4:
        confusion_types.append("low_calibration_reliability")
        actions.append("Use conservative reversible interventions only.")

    return {
        "confusion_flag": bool(confusion_types),
        "confusion_type": confusion_types or ["none"],
        "recommended_resolution_actions": actions or ["No confusion-triggered action required."],
    }
