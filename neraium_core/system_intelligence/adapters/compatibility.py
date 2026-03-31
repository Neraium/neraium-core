from __future__ import annotations

from typing import Any


def to_operator_compatibility(intel: dict[str, Any]) -> dict[str, Any]:
    transition = intel.get("transition_dynamics") or {}
    cf = intel.get("counterfactuals") or {}
    law = intel.get("law_engine_decision") or {}
    best = (cf.get("best_intervention") or {}).get("name", "monitor")
    regime = transition.get("regime", "unknown")
    law_note = str(law.get("law_layer_message") or "").strip()
    matched = list(law.get("matched_law_ids") or [])
    recommendation = f"Prioritize intervention scenario: {best}"
    if matched:
        recommendation = (
            f"{recommendation}. Law-layer matched {matched[0]} with bounded influence "
            f"(weight={law.get('law_influence_weight', 0.0)})."
        )
    if law_note:
        recommendation = f"{recommendation} {law_note}"
    return {
        "phase": regime,
        "trend": transition.get("transition_path", "stable"),
        "risk_level": "high" if float(transition.get("escalation_probability", 0.0)) > 0.7 else "medium" if float(transition.get("escalation_probability", 0.0)) > 0.45 else "low",
        "operational_recommendation": recommendation,
    }
