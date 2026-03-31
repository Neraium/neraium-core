from __future__ import annotations

from typing import Any


def to_operator_compatibility(intel: dict[str, Any]) -> dict[str, Any]:
    transition = intel.get("transition_dynamics") or {}
    cf = intel.get("counterfactuals") or {}
    best = (cf.get("best_intervention") or {}).get("name", "monitor")
    regime = transition.get("regime", "unknown")
    return {
        "phase": regime,
        "trend": transition.get("transition_path", "stable"),
        "risk_level": "high" if float(transition.get("escalation_probability", 0.0)) > 0.7 else "medium" if float(transition.get("escalation_probability", 0.0)) > 0.45 else "low",
        "operational_recommendation": f"Prioritize intervention scenario: {best}",
    }
