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
    recommendation = dict(intervention.get("recommendation") or {})
    reliability = intel.get("reliability_intelligence") or {}
    calibrated_rec = ((reliability.get("intervention_recommendation") or {}).get("recommendation_calibrated_confidence"))
    rec_confidence = float(calibrated_rec if calibrated_rec is not None else best_ranked.get("confidence", 0.0))
    reliability_warnings = list((((reliability.get("intervention_recommendation") or {}).get("reliability_trace") or {}).get("warnings") or []))
    trajectory = intel.get("trajectory_archetypes") or intel.get("trajectory_intelligence") or {}
    novelty = float(trajectory.get("novelty_score", 0.0) or 0.0)
    support_count = int(trajectory.get("support", trajectory.get("support_count", 0)) or 0)
    drift_warning = bool(((intel.get("reliability_intelligence") or {}).get("risk_advisory") or {}).get("drift_warning", False))
    structural_uncertainty = dict((intervention.get("structural_uncertainty_mode") or {}))
    override_applied = bool(recommendation.get("override_applied", False) or structural_uncertainty.get("override_applied", False))
    forced_posture = str(
        recommendation.get("recommended_posture")
        or structural_uncertainty.get("recommended_posture")
        or ("human_review_required" if bool(structural_uncertainty.get("active", False)) else "standard_advisory")
    )
    original_top_intervention = str(recommendation.get("original_top_intervention") or structural_uncertainty.get("original_top_intervention") or rec_name)
    fallback_reasons: list[str] = []
    if novelty >= 0.75:
        fallback_reasons.append("high_novelty")
    if support_count <= 2:
        fallback_reasons.append("sparse_support")
    if drift_warning:
        fallback_reasons.append("drift_warning")
    if reliability_warnings:
        fallback_reasons.append("reliability_warning")
    fallback_triggered = bool(fallback_reasons)

    law_note = str(law.get("law_layer_message") or "").strip()
    matched = list(law.get("matched_law_ids") or [])
    law_weight = float(law.get("law_influence_weight", 0.0))

    advisory_text = "Evidence limited; continue conservative monitoring and inspect active anomalies."
    if rec_confidence >= 0.68:
        advisory_text = f"Advisory focus: {rec_name}."
    elif rec_confidence >= 0.5:
        advisory_text = f"Conditional advisory focus: {rec_name} (reliability-limited)."

    if matched:
        advisory_text = f"{advisory_text} Law-layer matched {matched[0]} (bounded weight={law_weight:.2f})."
    if law_note:
        advisory_text = f"{advisory_text} {law_note}"

    if reliability_warnings:
        advisory_text = f"{advisory_text} Reliability notes: {reliability_warnings[0]}"
    if fallback_triggered:
        advisory_text = "Fallback active: insufficient trusted evidence for aggressive intervention; continue monitoring."
    if bool(structural_uncertainty.get("active", False)):
        advisory_text = "Structural uncertainty mode active: human review required; keep posture bounded and monitoring-first."
    if override_applied:
        advisory_text = "Structural uncertainty override applied: intervention forced to monitor; human review is required."

    return {
        "phase": str(transition.get("regime", "unknown")),
        "trend": str(transition.get("transition_path", "stable")),
        "risk_level": _risk_level(float(transition.get("escalation_probability", 0.0))),
        "operational_recommendation": advisory_text,
        "confidence_note": (
            "Intervention-focused advisory enabled only with stronger calibrated support/confidence."
            if rec_confidence < 0.68
            else "Intervention advisory reflects bounded evidence and remains operator-discretionary."
        ),
        "fallback_triggered": fallback_triggered,
        "fallback_reasons": sorted(set(fallback_reasons)),
        "recommended_intervention": rec_name,
        "forced_posture": forced_posture,
        "intervention_overridden": override_applied,
        "original_top_intervention": original_top_intervention,
        "structural_uncertainty_mode": structural_uncertainty,
    }
