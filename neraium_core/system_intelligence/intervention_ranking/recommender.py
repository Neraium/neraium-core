from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class InterventionRecommendationRanker:
    """Ranks interventions for operator support with bounded, explainable scoring."""

    def rank(
        self,
        *,
        candidates: list[dict[str, Any]],
        scored: dict[str, dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        ranked: list[dict[str, Any]] = []
        law_candidates = set(context.get("law_candidates") or [])
        family = str(context.get("trajectory_family", "unknown"))
        escalation_probability = _clip01(float(context.get("escalation_probability", 0.0) or 0.0))
        distance_to_critical = _clip01(float(context.get("distance_to_critical_region", 1.0) or 1.0))
        reversibility = _clip01(float(context.get("reversibility_score", 0.0) or 0.0))
        intervention_pressure = _clip01(0.65 * escalation_probability + 0.25 * (1.0 - distance_to_critical) + 0.10 * (1.0 - reversibility))

        for cand in candidates:
            name = str(cand["name"])
            score = scored[name]
            expected = float(score["intervention_effectiveness"])
            support = float(score["support"])
            uncertainty = float(score["uncertainty"])
            novelty = float(score["novelty_penalty"])
            context_match = float(score["context_match_quality"])
            worsening_signal = float(score.get("worsening_signal", 0.0))
            memory_weight = float(score.get("memory_effect_weight", 0.0))
            harmful_strength = float(score.get("harmful_signal_strength", worsening_signal))
            correlation_penalty = float(score.get("correlation_trap_penalty", 0.0))

            consistent_with_laws = 1.0 if any(term in name for term in law_candidates) else 0.6
            family_consistency = 1.0 if family in {"escalating", "reversible", "drift", "stable"} else 0.7
            intervention_activation = _clip01(0.35 + 0.65 * intervention_pressure)

            composite = (
                0.38 * expected
                + 0.20 * min(1.0, support / 6.0)
                + 0.12 * context_match
                + 0.08 * memory_weight
                + 0.12 * consistent_with_laws
                + 0.08 * family_consistency
                - 0.14 * uncertainty
                - 0.10 * novelty
                - 0.16 * harmful_strength
                - 0.14 * correlation_penalty
            )
            composite *= intervention_activation
            confidence = self._final_confidence(
                composite=composite,
                support=support,
                harmful_strength=harmful_strength,
                novelty=float(context.get("novelty_score", 0.0) or 0.0),
                reliability=float(context.get("calibration_reliability", 1.0) or 1.0),
                drift_warning=bool(context.get("drift_warning", False)),
            )
            memory_ablated = self._final_confidence(
                composite=composite - 0.08 * memory_weight,
                support=support,
                harmful_strength=harmful_strength,
                novelty=float(context.get("novelty_score", 0.0) or 0.0),
                reliability=float(context.get("calibration_reliability", 1.0) or 1.0),
                drift_warning=bool(context.get("drift_warning", False)),
            )
            confidence_info = self.compute_final_confidence(
                composite=composite,
                support=support,
                context_match=context_match,
                harmful_strength=harmful_strength,
                novelty=novelty,
                reliability=float(context.get("calibration_reliability", 1.0)),
                drift_warning=bool(context.get("drift_warning", False)),
            )
            confidence = confidence_info["confidence"]

            ranked.append(
                {
                    "name": name,
                    "intervention_type": cand["intervention_type"],
                    "intervention_target": cand["intervention_target"],
                    "rank_score": round(confidence, 4),
                    "composite_score": round(composite, 6),
                    "confidence": round(confidence, 4),
                    "confidence_regime": confidence_info["confidence_regime"],
                    "confidence_pre_calibration": round(confidence_info["confidence_pre_calibration"], 4),
                    "confidence_post_calibration": round(confidence_info["confidence_post_calibration"], 4),
                    "rationale": "Advisory ranking from bounded evidence and projection signals; not proof of causal effect.",
                    "ranking_factors": {
                        "expected_effectiveness": round(expected, 4),
                        "intervention_pressure": round(intervention_pressure, 4),
                        "intervention_activation": round(intervention_activation, 4),
                        "support": int(support),
                        "uncertainty": round(uncertainty, 4),
                        "novelty_penalty": round(novelty, 4),
                        "worsening_signal": round(worsening_signal, 4),
                        "harmful_signal_strength": round(harmful_strength, 4),
                        "context_match": round(context_match, 4),
                        "memory_effect_weight": round(memory_weight, 4),
                        "correlation_trap_penalty": round(correlation_penalty, 4),
                    },
                    "memory_ablation": {
                        "confidence_with_memory": round(confidence, 4),
                        "confidence_without_memory": round(memory_ablated, 4),
                        "confidence_delta": round(confidence - memory_ablated, 6),
                    },
                    "memory_ablation": {
                        "confidence_with_memory": round(confidence, 4),
                        "confidence_without_memory": round(memory_ablated, 4),
                        "confidence_delta": round(confidence - memory_ablated, 6),
                    },
                    "evidence_sources": {
                        "model_based_projection": cand.get("model_projection") or {},
                        "historical_evidence": score,
                    },
                    "warnings": [
                        "correlation_trap_risk_penalized"
                    ]
                    if correlation_penalty >= 0.45
                    else [],
                }
            )

        ranked.sort(key=lambda x: float(x["rank_score"]), reverse=True)
        best = ranked[0] if ranked else None
        best_without_memory = None
        if ranked:
            best_without_memory = max(
                ranked,
                key=lambda row: float((row.get("memory_ablation") or {}).get("confidence_without_memory", 0.0)),
            )

        for idx, item in enumerate(ranked):
            lower_reasons: list[str] = []
            for other in ranked[:idx]:
                if float(other["rank_score"]) <= float(item["rank_score"]):
                    continue
                lower_reasons.append(
                    f"Lower than {other['name']} due to weaker support/effectiveness or higher uncertainty/novelty."
                )
            item["why_ranked_lower"] = lower_reasons[:2]

        best_conf = float(best["confidence"]) if best else 0.0
        no_intervention_recommended = False
        if best:
            threshold = 0.48 if intervention_pressure < 0.35 else 0.40
            support_signal = _clip01(float(context.get("support_count", 0) or 0) / 8.0)
            abstention_eligible = intervention_pressure < 0.22 and support_signal >= 0.25
            if best_conf < threshold and abstention_eligible:
                no_intervention_recommended = True
                abstention_confidence = _clip01(0.78 + 0.16 * (1.0 - intervention_pressure) + 0.06 * support_signal)
                best_conf = max(best_conf, abstention_confidence)
                best["confidence"] = round(best_conf, 4)
                best["rank_score"] = round(best_conf, 4)
        best_delta = float((best or {}).get("memory_ablation", {}).get("confidence_delta", 0.0))
        contribution_status = "available" if ranked else "insufficient_evidence"
        return {
            "ranked_interventions": ranked,
            "best_intervention": best,
            "confidence": round(best_conf, 4),
            "recommendation_confidence": round(best_conf, 4),
            "no_intervention_recommended": no_intervention_recommended,
            "intervention_memory_contribution": {
                "status": contribution_status,
                "method": "candidate_score_ablation_delta",
                "best_confidence_delta": round(best_delta, 6),
                "best_choice_with_memory": (best or {}).get("name"),
                "best_choice_without_memory": (best_without_memory or {}).get("name"),
                "choice_changed_without_memory": bool(best and best_without_memory and best["name"] != best_without_memory["name"]),
            },
            "decision_trace": {
                "confidence_regime": (best or {}).get("confidence_regime"),
                "confidence_pre_calibration": (best or {}).get("confidence_pre_calibration"),
                "confidence_post_calibration": (best or {}).get("confidence_post_calibration"),
                "confidence_contributions": {
                    "historical_evidence": "primary when support and context match are strong",
                "counterfactual_projection": "secondary projection input",
                "law_and_trajectory_consistency": "consistency signal",
                "uncertainty_and_novelty_penalties": "always applied to keep recommendations conservative",
                "intervention_memory_weighting": "increases with repeated, context-matched supportive outcomes and decreases sharply under harmful history",
                "counterfactual_specificity_guard": "penalizes interventions that track generic regime improvement or already-stabilizing contexts",
                "dual_regime_confidence": "normal regime calibrates confidence; uncertain regime caps confidence for safety",
            },
                "assumptions_remaining": [
                    "Historical similarity may omit hidden confounders.",
                    "Counterfactual projections are approximate structural simulations.",
                    "Recommendations are advisory and require operator judgement.",
                ],
            },
        }

    @staticmethod
    def _final_confidence(*, composite: float, support: float, harmful_strength: float) -> float:
        confidence = _clip01(composite)
        if support < 2:
            confidence *= 0.8
        if harmful_strength > 0.05:
            confidence *= 0.62
        return confidence
    @staticmethod
    def calibrate_confidence(*, raw_confidence: float, regime: str) -> float:
        raw = _clip01(raw_confidence)
        if regime == "uncertain":
            return round(min(0.2, 0.015 + 0.145 * raw), 6)
        calibrated = 0.02 + 0.30 * (raw**20.0)
        return round(min(0.95, _clip01(calibrated)), 6)

    def compute_final_confidence(
        self,
        *,
        composite: float,
        support: float,
        context_match: float,
        harmful_strength: float,
        novelty: float,
        reliability: float,
        drift_warning: bool,
    ) -> dict[str, float | str]:
        pre_calibration = _clip01(composite)
        uncertain = bool(
            novelty >= 0.7
            or support <= 2
            or reliability < 0.5
            or drift_warning
            or harmful_strength >= 0.28
        )
        regime = "uncertain" if uncertain else "normal"

        post_calibration = self.calibrate_confidence(raw_confidence=pre_calibration, regime=regime)
        support_strength = _clip01(support / 4.0)
        context_strength = _clip01(context_match)
        evidence_strength = _clip01(0.6 * support_strength + 0.4 * context_strength)
        if regime == "uncertain":
            confidence = min(0.2, post_calibration + 0.03 * evidence_strength)
        else:
            confidence = min(0.95, post_calibration + 0.05 * evidence_strength)
        return {
            "confidence": float(confidence),
            "confidence_regime": regime,
            "confidence_pre_calibration": float(pre_calibration),
            "confidence_post_calibration": float(post_calibration),
        }

    def _final_confidence(
        self,
        *,
        composite: float,
        support: float,
        harmful_strength: float,
        novelty: float,
        reliability: float,
        drift_warning: bool,
    ) -> float:
        info = self.compute_final_confidence(
            composite=composite,
            support=support,
            context_match=0.0,
            harmful_strength=harmful_strength,
            novelty=novelty,
            reliability=reliability,
            drift_warning=drift_warning,
        )
        return float(info["confidence"])
