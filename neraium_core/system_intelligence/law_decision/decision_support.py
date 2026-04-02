from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _risk_from_level(level: str) -> float:
    return {"low": 0.25, "medium": 0.58, "high": 0.86}.get(str(level).strip().lower(), 0.35)


def _law_direction(law: dict[str, Any]) -> float:
    outcome = str(law.get("outcome") or "").lower()
    effect = _f(law.get("estimated_effect"), 0.0)
    if "recovery" in outcome or "decrease" in outcome:
        return -abs(effect)
    return abs(effect)


def apply_law_decision_support(
    *,
    gating: dict[str, Any],
    risk_assessment: dict[str, Any],
    operator_guidance: dict[str, Any],
    counterfactuals: dict[str, Any],
    trajectory_info: dict[str, Any],
) -> dict[str, Any]:
    eligible = list(gating.get("decision_eligible_laws") or [])
    suppressed = list(gating.get("suppressed_laws") or [])
    base_risk = _clamp01(
        0.55 * _f(risk_assessment.get("projected_score"), _risk_from_level(str(risk_assessment.get("current_risk_level", "low"))))
        + 0.45 * _risk_from_level(str(risk_assessment.get("current_risk_level", "low")))
    )

    novelty = _clamp01(_f(trajectory_info.get("novelty_score"), 1.0))
    conflict_penalty = 0.0
    if len(eligible) >= 2:
        dirs = [_law_direction(item) for item in eligible[:3]]
        if any(d > 0 for d in dirs) and any(d < 0 for d in dirs):
            conflict_penalty = 0.25

    total_weight = 0.0
    signed_effect = 0.0
    matched_ids: list[str] = []
    weights: dict[str, float] = {}
    recommendation_additions: list[str] = []
    intervention_links: list[dict[str, Any]] = []
    rationale_lines: list[str] = []

    best_intervention = (counterfactuals.get("best_intervention") or {}).get("name")
    for law in eligible[:3]:
        lid = str(law.get("law_id"))
        matched_ids.append(lid)
        conf = _clamp01(_f(law.get("effective_law_confidence"), 0.0))
        gen = _clamp01(_f(law.get("law_generalization_confidence"), 0.0))
        support = _f(law.get("support"), 0.0)
        traj_match = _clamp01(_f(law.get("trajectory_family_match"), 0.0))
        weight = _clamp01(0.55 * conf + 0.25 * gen + 0.10 * traj_match + 0.10 * min(1.0, support / 12.0))
        direction = _law_direction(law)
        signed_effect += weight * direction
        total_weight += weight
        weights[lid] = round(weight, 4)

        mechanism = str(law.get("mechanism") or "recurring mechanism")
        recommendation_additions.append(
            f"Law-layer match {lid} ({mechanism}) supports targeted inspection; applicability came from trajectory-family alignment with remaining uncertainty in generalization."
        )
        rationale_lines.append(
            f"{lid} eligible via support={int(support)} and effective_law_confidence={conf:.2f}; weight={weight:.2f}."
        )
        intervention_links.append(
            {
                "law_id": lid,
                "linked_mechanism": mechanism,
                "prioritized_intervention": best_intervention,
                "law_linked_intervention_rationale": "Intervention prioritization elevated because the matched candidate law repeatedly implicated this mechanism/subsystem under similar trajectory families.",
                "expected_relevance": round(_clamp01(0.4 + 0.6 * weight), 4),
                "confidence": round(_clamp01(0.5 * conf + 0.5 * gen), 4),
                "disclaimer": "Law linkage is decision-support evidence only and does not establish formal causality.",
            }
        )

    avg_weight = _clamp01(total_weight / max(1.0, min(3.0, float(len(eligible)))))
    law_influence_weight = _clamp01(min(0.35, avg_weight * (1.0 - 0.65 * novelty) * (1.0 - conflict_penalty)))
    law_risk_delta = max(-0.12, min(0.12, signed_effect * law_influence_weight))
    law_adjusted_risk = _clamp01(base_risk + law_risk_delta)
    confidence_adjustment = round(float(law_influence_weight * (0.25 if law_risk_delta >= 0 else 0.15)), 4)

    baseline_actions = list(operator_guidance.get("recommended_actions") or [])
    if recommendation_additions:
        baseline_actions = recommendation_additions[:1] + baseline_actions
    if not eligible:
        rationale_lines.append("Law-layer evidence was weak/novel or insufficiently supported; baseline guidance retained.")

    return {
        "base_risk": round(base_risk, 4),
        "law_adjusted_risk": round(law_adjusted_risk, 4),
        "law_risk_delta": round(law_risk_delta, 4),
        "law_influence_weight": round(law_influence_weight, 4),
        "confidence_adjustment": confidence_adjustment,
        "matched_law_ids": matched_ids,
        "influence_weights": weights,
        "suppressed_candidates": [str(item.get("law_id")) for item in suppressed],
        "law_decision_trace": {
            "matched_law_ids": matched_ids,
            "influence_weights": weights,
            "suppressed_candidates": [str(item.get("law_id")) for item in suppressed],
            "textual_rationale": " ".join(rationale_lines),
        },
        "law_informed_recommendations": baseline_actions,
        "law_linked_interventions": intervention_links,
        "law_layer_message": (
            "Law-layer evidence is influencing recommendations in a bounded way."
            if eligible
            else "Law-layer evidence is weak/novel, so recommendations remain primarily baseline-driven."
        ),
    }
