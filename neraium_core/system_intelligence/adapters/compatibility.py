from __future__ import annotations

from typing import Any


def _risk_level(escalation_probability: float) -> str:
    if escalation_probability > 0.7:
        return "high"
    if escalation_probability > 0.45:
        return "medium"
    return "low"


def to_operator_compatibility(intel: dict[str, Any]) -> dict[str, Any]:
    """Build conservative legacy-facing recommendation text from bounded intelligence output."""
    transition = intel.get("transition_dynamics") or {}
    cf = intel.get("counterfactuals") or {}
    law = intel.get("law_engine_decision") or {}
    intervention = intel.get("intervention_intelligence") or {}

    best_cf_name = str((cf.get("best_intervention") or {}).get("name", "monitor"))
    best_ranked = ((intervention.get("recommendation") or {}).get("best_intervention") or {})
    rec_name = str(best_ranked.get("name", best_cf_name))
    rec_confidence = float(best_ranked.get("confidence", 0.0))

    law_note = str(law.get("law_layer_message") or "").strip()
    matched = list(law.get("matched_law_ids") or [])
    law_weight = float(law.get("law_influence_weight", 0.0))

    advisory_text = "Evidence limited; continue conservative monitoring and inspect active anomalies."
    if rec_confidence >= 0.6:
        advisory_text = f"Advisory focus: {rec_name}."

    if matched:
        advisory_text = f"{advisory_text} Law-layer matched {matched[0]} (bounded weight={law_weight:.2f})."
    if law_note:
        advisory_text = f"{advisory_text} {law_note}"

    return {
        "phase": str(transition.get("regime", "unknown")),
        "trend": str(transition.get("transition_path", "stable")),
        "risk_level": _risk_level(float(transition.get("escalation_probability", 0.0))),
        "operational_recommendation": advisory_text,
        "confidence_note": (
            "Intervention-focused advisory enabled only with stronger support/confidence."
            if rec_confidence < 0.6
            else "Intervention advisory reflects bounded evidence and remains operator-discretionary."
        ),
    }
