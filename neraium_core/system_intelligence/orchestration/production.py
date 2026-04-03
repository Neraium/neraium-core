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
    """Minimal production-safe path for deployable operator-facing intelligence."""

    def __init__(self) -> None:
        self.latent = LatentStructuralStateEncoder(latent_dim=3)
        self.transitions = LatentTransitionModel()
        self.counterfactuals = CounterfactualInterventionEngine()
        self.archetypes = StructuralArchetypeMemory()
        self.trajectory_memory = CrossSystemTrajectoryMemory(window_size=10)
        self.trajectory_forecast = TrajectoryConditionedForecaster()
        self.intervention_intelligence = InterventionIntelligenceEngine()
        self.reliability = StructuralReliabilityLayer()

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
        }
