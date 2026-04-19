from __future__ import annotations

from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def evaluate_law_candidates(
    *,
    law_candidates: list[dict[str, Any]],
    trajectory_info: dict[str, Any],
    mechanism_info: dict[str, Any],
    governance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate structural law candidates before they can influence decisions."""

    family = str(trajectory_info.get("current_trajectory_path_family") or "unknown")
    novelty = _clamp01(_f(trajectory_info.get("novelty_score"), 1.0))
    mechanism_assoc = {
        str(item.get("mechanism")): item
        for item in list(mechanism_info.get("mechanism_candidates") or [])
        if isinstance(item, dict)
    }

    governance = governance or {}
    governance_laws = {str(item.get("law_id")): item for item in list(governance.get("laws") or []) if isinstance(item, dict)}

    eligible: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []

    for row in law_candidates:
        law = dict(row)
        support = max(0.0, _f(law.get("support")))
        effect = _clamp01(abs(_f(law.get("estimated_effect"))))
        robustness = _clamp01(_f(law.get("robustness"), 0.0))
        confidence = _clamp01(_f(law.get("confidence"), 0.0))
        consistency = _clamp01(_f(law.get("consistency_across_assets_runs"), 0.0))
        counterfactual_consistency = _clamp01(_f(law.get("counterfactual_consistency"), 0.0))
        family_support = _clamp01(_f(law.get("trajectory_family_support"), 0.0))
        expected_family = str((law.get("applicability") or {}).get("trajectory_family", family) or family)
        family_match = 1.0 if expected_family in {"unknown", family} else 0.35

        mechanism = str(law.get("mechanism", ""))
        mech_item = mechanism_assoc.get(mechanism, {})
        if mechanism and mech_item:
            family_support = max(family_support, _clamp01(_f(mech_item.get("family_conditioned_predictive_value"), 0.0)))

        novelty_penalty = 0.55 * novelty
        support_score = _clamp01(min(1.0, support / 14.0))
        effective_conf = _clamp01(
            0.24 * support_score
            + 0.22 * effect
            + 0.16 * robustness
            + 0.12 * family_support
            + 0.12 * consistency
            + 0.08 * counterfactual_consistency
            + 0.06 * confidence
            - novelty_penalty
        )
        generalization_conf = _clamp01(0.45 * consistency + 0.35 * family_support + 0.20 * (1.0 - novelty))

        gov_entry = governance_laws.get(str(law.get("law_id")), {})
        law_stage = str(gov_entry.get("current_stage") or "observed_pattern")

        reasons: list[str] = []
        if support < 4:
            reasons.append("support_below_minimum")
        if effect < 0.12:
            reasons.append("effect_too_small")
        if robustness < 0.35:
            reasons.append("low_robustness")
        if family_match < 0.6:
            reasons.append("weak_trajectory_family_match")
        if novelty > 0.78:
            reasons.append("high_novelty_penalty")
        if effective_conf < 0.52:
            reasons.append("insufficient_effective_law_confidence")
        if law_stage != "decision_grade_heuristic":
            reasons.append("not_decision_grade_heuristic")

        gated = {
            **law,
            "effective_law_confidence": round(effective_conf, 4),
            "law_generalization_confidence": round(generalization_conf, 4),
            "trajectory_family_match": round(family_match, 4),
            "novelty_penalty": round(novelty_penalty, 4),
            "current_stage": law_stage,
            "stage_confidence": gov_entry.get("stage_confidence"),
            "promotion_eligibility": gov_entry.get("promotion_eligibility"),
            "promotion_blockers": list(gov_entry.get("promotion_blockers") or []),
            "risk_flags": list(gov_entry.get("risk_flags") or []),
        }

        if reasons:
            gated["suppression_reasons"] = reasons
            suppressed.append(gated)
        else:
            eligible.append(gated)

    eligible.sort(key=lambda item: (item["effective_law_confidence"], item.get("support", 0)), reverse=True)
    suppressed.sort(key=lambda item: (item["effective_law_confidence"], item.get("support", 0)), reverse=True)

    return {
        "decision_eligible_laws": eligible,
        "suppressed_laws": suppressed,
        "suppression_reasons": {str(item.get("law_id")): list(item.get("suppression_reasons", [])) for item in suppressed},
        "decision_grade_law_ids": [str(item.get("law_id")) for item in eligible],
    }
