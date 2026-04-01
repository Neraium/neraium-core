from __future__ import annotations

from typing import Any

from ..intervention_memory.memory import InterventionMemoryStore
from ..intervention_ranking.recommender import InterventionRecommendationRanker
from .scoring import InterventionEffectivenessScorer


_INTERVENTION_MAP = {
    "remove_top_driver_contribution": ("remove_or_suppress_top_driver", "top_driver"),
    "restore_relationship_cluster_to_baseline": ("restore_relationship_cluster", "relationship_cluster"),
    "suppress_subsystem_instability": ("damp_subsystem_instability", "subsystem"),
}


def _map_intervention(name: str, intervention_type: str | None = None, intervention_target: str | None = None) -> tuple[str, str]:
    if intervention_type and intervention_target:
        return str(intervention_type), str(intervention_target)
    mapped_type, mapped_target = _INTERVENTION_MAP.get(str(name), ("other", "system"))
    return str(intervention_type or mapped_type), str(intervention_target or mapped_target)


class InterventionIntelligenceEngine:
    def __init__(self) -> None:
        self.memory = InterventionMemoryStore()
        self.scorer = InterventionEffectivenessScorer(self.memory)
        self.ranker = InterventionRecommendationRanker()
        self._uncertainty_thresholds = {
            "novelty": 0.72,
            "low_similarity": 0.45,
            "support": 2,
            "calibration_reliability": 0.5,
            "transfer_mismatch": 0.62,
            "ambiguity": 0.12,
        }

    def update(
        self,
        *,
        asset_id: str,
        observation: dict[str, Any],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        mechanism: dict[str, Any],
        laws: dict[str, Any],
        counterfactuals: dict[str, Any],
    ) -> dict[str, Any]:
        pre_or_post_state = {
            "escalation_probability": float(transition.get("escalation_probability", 0.0)),
            "reversibility_score": float(transition.get("reversibility_score", 0.0)),
            "distance_to_critical_region": float(transition.get("distance_to_critical_region", 0.0)),
        }
        novelty_score = float(trajectory.get("novelty_score", 0.5))
        family_similarity = trajectory.get("family_similarity", trajectory.get("trajectory_similarity"))
        novelty_implied_similarity = max(0.0, min(1.0, 1.0 - novelty_score))
        if family_similarity is None:
            family_similarity = novelty_implied_similarity
        else:
            family_similarity = max(float(family_similarity), novelty_implied_similarity)
        support_count = trajectory.get("support", trajectory.get("support_count", 0))
        context = {
            "latent_state": list((observation.get("latent_embedding") or [])),
            "trajectory_family": str(trajectory.get("current_trajectory_path_family", trajectory.get("current_trajectory_family", "unknown"))),
            "transition_path": str(transition.get("transition_path", "unknown")),
            "regime": str(transition.get("regime", "unknown")),
            "escalation_probability": float(transition.get("escalation_probability", 0.0)),
            "distance_to_critical_region": float(transition.get("distance_to_critical_region", 1.0)),
            "reversibility_score": float(transition.get("reversibility_score", 0.5)),
            "mechanism_candidates": [str(m.get("mechanism", "")) for m in (mechanism.get("mechanism_candidates") or [])[:4]],
            "law_candidates": [str(l.get("law", "")) for l in (laws.get("law_candidates") or [])[:4]],
            "novelty_score": novelty_score,
            "family_similarity": float(family_similarity or 0.0),
            "support_count": int(support_count or 0),
            "drift_warning": bool(observation.get("drift_alert", observation.get("drift_warning", False))),
            "transfer_mismatch": float((observation.get("transfer_metrics") or {}).get("mismatch_penalty", 0.0) or 0.0),
            "calibration_reliability": float((observation.get("calibration") or {}).get("reliability", 1.0) or 1.0),
        }

        evidence_update = self.memory.finalize_if_ready(asset_id=asset_id, post_state=pre_or_post_state)

        intervention_observation = observation.get("applied_intervention") or observation.get("intervention_event")
        if isinstance(intervention_observation, dict):
            intervention_name = str(intervention_observation.get("name", intervention_observation.get("type", "other")))
            intervention_type, intervention_target = _map_intervention(
                intervention_name,
                intervention_type=intervention_observation.get("type"),
                intervention_target=intervention_observation.get("target"),
            )
            outcome_label = intervention_observation.get("outcome_label")
            if outcome_label is not None:
                self.memory.ingest_observed_outcome(
                    asset_id=asset_id,
                    intervention_type=intervention_type,
                    intervention_target=intervention_target,
                    context=context,
                    outcome_label=str(outcome_label),
                    state=pre_or_post_state,
                    metadata={
                        "source": "replay_observed_outcome",
                        "latent_state": list(context.get("latent_state") or []),
                    },
                )
            else:
                self.memory.register_intervention_start(
                    asset_id=asset_id,
                    intervention_type=intervention_type,
                    intervention_target=intervention_target,
                    context=context,
                    pre_state=pre_or_post_state,
                    metadata={
                        "source": "operator_input",
                        "confidence": float(intervention_observation.get("confidence", 0.5)),
                        "latent_state": list(context.get("latent_state") or []),
                    },
                )

        candidates = []
        for scenario in counterfactuals.get("scenario_rankings") or []:
            name = str(scenario.get("name", "other"))
            intervention_type, target = _INTERVENTION_MAP.get(name, ("other", str(scenario.get("leverage_component", "system"))))
            candidates.append(
                {
                    "name": name,
                    "intervention_type": intervention_type,
                    "intervention_target": target,
                    "model_projection": {
                        "expected_escalation_reduction": float(scenario.get("risk_delta", 0.0)),
                    },
                }
            )

        scored: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            scored[cand["name"]] = self.scorer.score_candidate(
                context=context,
                intervention_type=cand["intervention_type"],
                intervention_target=cand["intervention_target"],
                model_projection=cand.get("model_projection"),
            )

        ranked = self.ranker.rank(candidates=candidates, scored=scored, context=context)
        memory_support = max((int((scored.get(c.get("name", "")) or {}).get("support", 0)) for c in candidates), default=0)
        memory_summary_support = int((self.memory.support_summary() or {}).get("support_count", 0) or 0)
        memory_match_quality = max(
            (float((scored.get(c.get("name", "")) or {}).get("context_match_quality", 0.0)) for c in candidates),
            default=0.0,
        )
        context["support_count"] = max(int(context.get("support_count", 0) or 0), memory_support, memory_summary_support)
        ranked = self._apply_conservative_fallback(ranked=ranked, context=context)
        recommendation_confidence = float(ranked.get("recommendation_confidence", 0.0))

        return {
            "status": "active",
            "evidence_update": evidence_update or {"status": "no_new_record"},
            "context": context,
            "historical_evidence": {
                "support_summary": self.memory.support_summary(),
                "effectiveness_by_intervention": scored,
            },
            "model_based_projection": {
                "source": "counterfactual_engine",
                "scenario_rankings": counterfactuals.get("scenario_rankings") or [],
            },
            "recommendation": {
                **ranked,
                "advisory": True,
                "disclaimer": "Intervention ranking is decision support only, based on bounded historical evidence and model projections.",
                "memory_match_count": memory_support,
                "memory_match_quality": round(memory_match_quality, 4),
            },
            "structural_uncertainty_mode": self._structural_uncertainty_mode(ranked=ranked, context=context),
            "uncertainty_summary": {
                "confidence": round(recommendation_confidence, 4),
                "confidence_band": "high" if recommendation_confidence >= 0.7 else "moderate" if recommendation_confidence >= 0.45 else "low",
                "assumptions": [
                    "Historical evidence is observational and context-conditioned.",
                    "Model projections are approximate and not formal causal proof.",
                ],
            },
        }

    def _structural_uncertainty_mode(self, *, ranked: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        novelty = float(context.get("novelty_score", 0.0))
        similarity = float(context.get("family_similarity", 0.0))
        support = int(context.get("support_count", 0) or 0)
        reliability = float(context.get("calibration_reliability", 1.0))
        mismatch = float(context.get("transfer_mismatch", 0.0))
        reasons = list(ranked.get("fallback_reasons") or [])
        weak_support = support <= self._uncertainty_thresholds["support"]
        elevated_context_risk = bool(
            novelty >= self._uncertainty_thresholds["novelty"]
            or similarity <= self._uncertainty_thresholds["low_similarity"]
            or reliability < self._uncertainty_thresholds["calibration_reliability"]
            or mismatch >= self._uncertainty_thresholds["transfer_mismatch"]
            or bool(reasons)
        )
        active = bool(elevated_context_risk or (weak_support and bool(reasons)))
        reason = "stable_context"
        if active:
            reason = ",".join(reasons[:3]) if reasons else "low_trust_structural_context"
        return {
            "active": active,
            "reason": reason,
            "novelty": round(novelty, 6),
            "support": support,
            "reliability": round(reliability, 6),
            "recommended_posture": "human_review_required" if active else "standard_advisory",
        }

    def _apply_conservative_fallback(self, *, ranked: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        novelty = float(context.get("novelty_score", 0.0))
        similarity = float(context.get("family_similarity", 0.0))
        support = int(context.get("support_count", 0) or 0)
        drift_warning = bool(context.get("drift_warning", False))
        mismatch = float(context.get("transfer_mismatch", 0.0))
        reliability = float(context.get("calibration_reliability", 1.0))
        scores = [float(r.get("confidence", 0.0)) for r in list(ranked.get("ranked_interventions") or [])[:2]]
        ambiguity = abs(scores[0] - scores[1]) if len(scores) >= 2 else 1.0
        reasons: list[str] = []
        if novelty >= self._uncertainty_thresholds["novelty"]:
            reasons.append("high_novelty")
        if similarity <= self._uncertainty_thresholds["low_similarity"]:
            reasons.append("low_structural_similarity")
        weak_support = support <= self._uncertainty_thresholds["support"]
        elevated_context_risk = (
            novelty >= self._uncertainty_thresholds["novelty"]
            or similarity <= self._uncertainty_thresholds["low_similarity"]
            or drift_warning
            or mismatch >= self._uncertainty_thresholds["transfer_mismatch"]
            or reliability < self._uncertainty_thresholds["calibration_reliability"]
        )
        if weak_support and elevated_context_risk:
            reasons.append("weak_support")
        if drift_warning:
            reasons.append("drift_warning")
        if mismatch >= self._uncertainty_thresholds["transfer_mismatch"]:
            reasons.append("transfer_mismatch")
        if reliability < self._uncertainty_thresholds["calibration_reliability"]:
            reasons.append("weak_calibration_reliability")
        if (
            ambiguity <= self._uncertainty_thresholds["ambiguity"]
            and min(scores) >= 0.2
            and (support > self._uncertainty_thresholds["support"] or elevated_context_risk)
        ):
            reasons.append("competing_explanations")

        high_novelty_weak_support = (
            (novelty >= self._uncertainty_thresholds["novelty"] or similarity <= self._uncertainty_thresholds["low_similarity"])
            and support <= self._uncertainty_thresholds["support"]
        )
        if reasons:
            bounded_conf = min(float(ranked.get("confidence", 0.0)), 0.52)
            if high_novelty_weak_support or drift_warning:
                bounded_conf = min(bounded_conf, 0.35)
            ranked["confidence"] = round(bounded_conf, 4)
            ranked["recommendation_confidence"] = round(bounded_conf, 4)
            ranked["fallback_triggered"] = True
            ranked["fallback_reasons"] = sorted(set(reasons))
            ranked["recommended_posture"] = "human_review_required" if high_novelty_weak_support or drift_warning else "bounded_advisory"
            if high_novelty_weak_support or drift_warning:
                ranked["best_intervention"] = {
                    "name": "monitor",
                    "intervention_type": "monitor",
                    "intervention_target": "system",
                    "confidence": round(bounded_conf, 4),
                    "rank_score": round(bounded_conf, 4),
                    "rationale": "Insufficient trusted structural evidence for aggressive intervention; escalate to human review.",
                }
            for item in list(ranked.get("ranked_interventions") or []):
                item["confidence"] = round(min(float(item.get("confidence", 0.0)), bounded_conf), 4)
                item.setdefault("warnings", [])
                item["warnings"] = sorted(set([*item["warnings"], "fallback_conservative_bounding"]))
        else:
            ranked["fallback_triggered"] = False
            ranked["fallback_reasons"] = []
            ranked["recommended_posture"] = "standard_advisory"
        return ranked
