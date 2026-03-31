from __future__ import annotations

from typing import Any

from ..archetypes.archetype_memory import StructuralArchetypeMemory
from ..counterfactuals.intervention_engine import CounterfactualInterventionEngine
from ..forecast.trajectory_conditioned import TrajectoryConditionedForecaster
from ..intervention_intelligence.engine import InterventionIntelligenceEngine
from ..reliability import StructuralReliabilityLayer
from ..structural_state.latent_state import LatentStructuralStateEncoder
from ..trajectory_memory.memory import CrossSystemTrajectoryMemory
from ..transition_model.transition_dynamics import LatentTransitionModel


class ProductionIntelligenceOrchestrator:
    """Minimal production-safe path for deployable operator-facing intelligence.

    Production boundary (decision-grade):
    - latent_structural_state
    - transition_dynamics
    - trajectory_archetypes / trajectory_forecast
    - intervention_intelligence
    - reliability_intelligence
    Advisory/experimental layers are intentionally excluded from this class.
    """

    def __init__(self) -> None:
        self.latent = LatentStructuralStateEncoder(latent_dim=3)
        self.transitions = LatentTransitionModel()
        self.counterfactuals = CounterfactualInterventionEngine()
        self.archetypes = StructuralArchetypeMemory()
        self.trajectory_memory = CrossSystemTrajectoryMemory(window_size=10)
        self.trajectory_forecast = TrajectoryConditionedForecaster()
        self.intervention_intelligence = InterventionIntelligenceEngine()
        self.reliability = StructuralReliabilityLayer()

    @staticmethod
    def _clip01(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _enforce_output_confidence(
        cls,
        *,
        recommendation: dict[str, Any],
        structural_uncertainty: dict[str, Any],
        rec_cal: float,
        novelty: float,
        support_count: int,
        drift_warning: bool,
    ) -> None:
        best = recommendation.get("best_intervention")
        if not isinstance(best, dict):
            return

        uncertain = bool(structural_uncertainty.get("active", False) or novelty >= 0.7 or support_count <= 2 or drift_warning)
        regime = "uncertain" if uncertain else "normal"
        current_confidence = cls._clip01(best.get("confidence", 0.0))
        calibrated_candidate = max(
            cls._clip01(rec_cal),
            cls._clip01(best.get("confidence_post_calibration", 0.0)),
            current_confidence,
        )
        if regime == "uncertain":
            enforced_confidence = min(0.2, max(0.0, calibrated_candidate))
        else:
            enforced_confidence = min(0.95, max(0.6, 0.6 + 0.35 * calibrated_candidate))

        best["confidence"] = round(float(enforced_confidence), 6)
        best["confidence_regime"] = regime
        best["confidence_post_calibration"] = round(float(enforced_confidence), 6)
        recommendation["confidence_regime"] = regime
        recommendation["confidence_post_calibration"] = round(float(enforced_confidence), 6)

    def update(self, observation: dict[str, Any]) -> dict[str, Any]:
        latent_snapshot = self.latent.encode(observation)
        transition = self.transitions.assess(latent_snapshot.embedding)
        escalating = transition.transition_path in {"escalating"} or transition.escalation_probability > 0.65
        in_critical = transition.regime == "critical"

        counterfactuals = self.counterfactuals.evaluate(observation, transition_escalation=transition.escalation_probability)
        archetypes = self.archetypes.update(
            asset_id=str(observation.get("asset_id", "unknown")),
            embedding=latent_snapshot.embedding,
            escalating=escalating,
        )
        relationship_names = [
            f"{str(r.get('source', ''))}->{str(r.get('target', ''))}" for r in list(observation.get("top_relationships") or [])[:3]
        ]
        trajectory = self.trajectory_memory.update(
            asset_id=str(observation.get("asset_id", "unknown")),
            embedding=latent_snapshot.embedding,
            escalating=escalating,
            in_critical_region=in_critical,
            phase_shift=transition.transition_path in {"escalating", "reversible"},
            mechanism_names=relationship_names,
        )
        forecast = self.trajectory_forecast.forecast(
            trajectory_intelligence=trajectory,
            transition_dynamics={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
            },
        )
        intervention = self.intervention_intelligence.update(
            asset_id=str(observation.get("asset_id", "unknown")),
            observation={**observation, "latent_embedding": latent_snapshot.embedding},
            transition={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
                "reversibility_score": transition.reversibility_score,
                "distance_to_critical_region": transition.distance_to_critical_region,
            },
            trajectory=trajectory,
            mechanism={"mechanism_candidates": []},
            laws={"law_candidates": []},
            counterfactuals=counterfactuals,
        )

        asset_id = str(observation.get("asset_id", "unknown"))
        step = int(getattr(self.trajectory_memory, "step", 0))
        self.reliability.finalize_due(
            asset_id=asset_id,
            step=step,
            transition={
                "escalation_probability": transition.escalation_probability,
                "transition_path": transition.transition_path,
                "regime": transition.regime,
            },
        )
        reliability = self.reliability.calibrate_all(
            asset_id=asset_id,
            step=step,
            transition={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
                "uncertainty": transition.uncertainty,
            },
            trajectory_forecast=forecast,
            laws={"law_candidates": []},
            intervention=intervention,
            cross={},
        )

        rec_cal = float(((reliability.get("intervention_recommendation") or {}).get("recommendation_calibrated_confidence", 0.0)))
        if intervention.get("recommendation") and intervention["recommendation"].get("best_intervention"):
            best = intervention["recommendation"]["best_intervention"]
            best["raw_confidence"] = best.get("confidence", 0.0)
            best["confidence"] = round(rec_cal, 6)
        ranked = list((intervention.get("recommendation") or {}).get("ranked_interventions") or [])
        best_ranked = dict((intervention.get("recommendation") or {}).get("best_intervention") or {})
        reliability_trace = dict(((reliability.get("intervention_recommendation") or {}).get("reliability_trace") or {}))
        warnings = list(reliability_trace.get("warnings") or [])
        memory_summary = dict(((intervention.get("historical_evidence") or {}).get("support_summary") or {}))
        structural_uncertainty = dict(intervention.get("structural_uncertainty_mode") or {})
        recommendation = intervention.setdefault("recommendation", {})
        original_top_intervention = str(best_ranked.get("name", "monitor"))
        override_applied = False
        if bool(structural_uncertainty.get("active", False)):
            override_applied = True
            fallback_reasons = sorted(set([*list(recommendation.get("fallback_reasons") or []), "structural_uncertainty_mode"]))
            recommendation["fallback_triggered"] = True
            recommendation["fallback_reasons"] = fallback_reasons
            recommendation["recommended_posture"] = "human_review_required"
            recommendation["override_applied"] = True
            recommendation["override_reason"] = "structural_uncertainty"
            recommendation["original_top_intervention"] = original_top_intervention
            recommendation["best_intervention"] = {
                "name": "monitor",
                "intervention_type": "monitor",
                "intervention_target": "system",
                "confidence": round(min(rec_cal, 0.35), 6),
                "rank_score": round(min(rec_cal, 0.35), 6),
                "rationale": "Hard safety override: structural uncertainty mode active; require human review and monitoring posture.",
            }
            structural_uncertainty["recommended_posture"] = "human_review_required"
            structural_uncertainty["override_applied"] = True
            structural_uncertainty["override_reason"] = "structural_uncertainty"
            structural_uncertainty["original_top_intervention"] = original_top_intervention

        drift_warning = bool(
            ((intervention.get("context") or {}).get("drift_warning", False))
            or ((reliability.get("risk_advisory") or {}).get("drift_warning", False))
        )
        novelty = self._clip01((intervention.get("context") or {}).get("novelty_score", trajectory.get("novelty_score", 0.0)))
        context_support_count = int(((intervention.get("context") or {}).get("support_count", 0)) or 0)
        memory_support_count = int((memory_summary.get("support_count", 0)) or 0)
        support_count = max(context_support_count, memory_support_count)
        self._enforce_output_confidence(
            recommendation=recommendation,
            structural_uncertainty=structural_uncertainty,
            rec_cal=rec_cal,
            novelty=novelty,
            support_count=support_count,
            drift_warning=drift_warning,
        )

        fallback_policy = {
            "triggered": bool(recommendation.get("fallback_triggered", False)),
            "reasons": list(recommendation.get("fallback_reasons") or []),
            "high_novelty_weak_support": bool(novelty >= 0.72 and support_count <= 2),
            "posture": str(recommendation.get("recommended_posture", "standard_advisory")),
            "max_allowed_confidence": round(float(recommendation.get("recommendation_confidence", recommendation.get("confidence", 0.0))), 6),
        }
        ranked = list((intervention.get("recommendation") or {}).get("ranked_interventions") or [])
        best_ranked = dict((intervention.get("recommendation") or {}).get("best_intervention") or {})
        evidence_classes = {
            "helpful": len(list(memory_summary.get("helpful_interventions") or [])),
            "harmful": len(list(memory_summary.get("harmful_interventions") or [])),
            "neutral": len(list(memory_summary.get("neutral_interventions") or [])),
        }
        trace = {
            "state_context": {
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "trajectory_family": trajectory.get("current_trajectory_path_family", trajectory.get("current_trajectory_family", "unknown")),
                "escalation_probability": round(float(transition.escalation_probability), 6),
            },
            "top_intervention_factors": list((best_ranked.get("ranking_factors") or {}).items())[:5],
            "calibrated_confidence": round(rec_cal, 6),
            "intervention_history_evidence": {
                "support_count": support_count,
                "evidence_classes": evidence_classes,
            },
            "law_influence": {
                "status": "excluded_from_core_trace",
                "reason": "Law influence is advisory unless decision-grade governance is explicitly active.",
            },
            "warnings_or_blockers": warnings,
            "structural_uncertainty_mode": structural_uncertainty,
            "fallback_policy": fallback_policy,
            "override_applied": override_applied,
            "override_reason": "structural_uncertainty" if override_applied else None,
            "original_top_intervention": original_top_intervention,
            "original_candidate_interventions": [str(item.get("name", "unknown")) for item in ranked],
            "audit_note": "Trace includes only production decision-grade layers.",
        }

        return {
            "latent_structural_state": {
                "embedding": [round(float(v), 6) for v in latent_snapshot.embedding],
                "summary_features": {k: round(float(v), 6) for k, v in latent_snapshot.summary_features.items()},
                "trajectory": [[round(float(v), 6) for v in row] for row in latent_snapshot.trajectory],
                "velocity": round(float(latent_snapshot.velocity), 6),
                "acceleration": round(float(latent_snapshot.acceleration), 6),
            },
            "transition_dynamics": {
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": round(float(transition.escalation_probability), 6),
                "reversibility_score": round(float(transition.reversibility_score), 6),
                "distance_to_critical_region": round(float(transition.distance_to_critical_region), 6),
                "uncertainty": round(float(transition.uncertainty), 6),
            },
            "counterfactuals": counterfactuals,
            "archetype_intelligence": archetypes,
            "trajectory_archetypes": trajectory,
            "trajectory_forecast": forecast,
            "intervention_intelligence": intervention,
            "reliability_intelligence": reliability,
            "core_decision_trace": trace,
        }
