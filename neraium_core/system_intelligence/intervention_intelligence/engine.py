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


class InterventionIntelligenceEngine:
    def __init__(self) -> None:
        self.memory = InterventionMemoryStore()
        self.scorer = InterventionEffectivenessScorer(self.memory)
        self.ranker = InterventionRecommendationRanker()

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
        context = {
            "latent_state": list((observation.get("latent_embedding") or [])),
            "trajectory_family": str(trajectory.get("current_trajectory_path_family", trajectory.get("current_trajectory_family", "unknown"))),
            "transition_path": str(transition.get("transition_path", "unknown")),
            "regime": str(transition.get("regime", "unknown")),
            "mechanism_candidates": [str(m.get("mechanism", "")) for m in (mechanism.get("mechanism_candidates") or [])[:4]],
            "law_candidates": [str(l.get("law", "")) for l in (laws.get("law_candidates") or [])[:4]],
            "novelty_score": float(trajectory.get("novelty_score", 0.5)),
        }

        evidence_update = self.memory.finalize_if_ready(asset_id=asset_id, post_state=pre_or_post_state)

        intervention_observation = observation.get("applied_intervention") or observation.get("intervention_event")
        if isinstance(intervention_observation, dict):
            self.memory.register_intervention_start(
                asset_id=asset_id,
                intervention_type=str(intervention_observation.get("type", "other")),
                intervention_target=str(intervention_observation.get("target", "system")),
                context=context,
                pre_state=pre_or_post_state,
                metadata={"source": "operator_input", "confidence": float(intervention_observation.get("confidence", 0.5))},
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
            },
            "uncertainty_summary": {
                "confidence": round(recommendation_confidence, 4),
                "confidence_band": "high" if recommendation_confidence >= 0.7 else "moderate" if recommendation_confidence >= 0.45 else "low",
                "assumptions": [
                    "Historical evidence is observational and context-conditioned.",
                    "Model projections are approximate and not formal causal proof.",
                ],
            },
        }
