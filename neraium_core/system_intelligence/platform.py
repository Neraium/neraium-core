from __future__ import annotations

from typing import Any

from .adapters.compatibility import to_operator_compatibility
from .archetypes.archetype_memory import StructuralArchetypeMemory
from .counterfactuals.intervention_engine import CounterfactualInterventionEngine
from .forecast.trajectory_conditioned import TrajectoryConditionedForecaster
from .law_engine import StructuralLawDecisionEngine
from .law_extraction.extractor import StructuralLawExtractor
from .mechanisms.discovery import MechanismDiscoveryLayer
from .intervention_intelligence.engine import InterventionIntelligenceEngine
from .structural_state.latent_state import LatentStructuralStateEncoder
from .trajectory_memory.memory import CrossSystemTrajectoryMemory
from .transition_model.transition_dynamics import LatentTransitionModel


class StructuralSystemIntelligencePlatform:
    """Integrated structural intelligence stack for state, trajectories, interventions, archetypes, mechanisms."""

    def __init__(self) -> None:
        self.latent = LatentStructuralStateEncoder(latent_dim=3)
        self.transitions = LatentTransitionModel()
        self.counterfactuals = CounterfactualInterventionEngine()
        self.archetypes = StructuralArchetypeMemory()
        self.trajectory_memory = CrossSystemTrajectoryMemory(window_size=10)
        self.trajectory_forecast = TrajectoryConditionedForecaster()
        self.mechanisms = MechanismDiscoveryLayer()
        self.law_extractor = StructuralLawExtractor()
if rec_confidence > 0.6:
    operational_recommendation = f"Advisory focus: {rec_name}"
else:
    operational_recommendation = "Continue monitoring system behavior and investigate anomalies."

return {
    "phase": regime,
    "trend": transition.get("trend"),
    "risk_level": "high" if transition.get("escalation_probability", 0) > 0.7 else "moderate",
    "operational_recommendation": operational_recommendation,
}

    def update(self, observation: dict[str, Any]) -> dict[str, Any]:
        latent_snapshot = self.latent.encode(observation)
        transition = self.transitions.assess(latent_snapshot.embedding)
        escalating = transition.transition_path in {"escalating"} or transition.escalation_probability > 0.65
        recovering = transition.transition_path in {"reversible"}
        in_critical = transition.regime == "critical"

        cf = self.counterfactuals.evaluate(observation, transition_escalation=transition.escalation_probability)
        arch = self.archetypes.update(
            asset_id=str(observation.get("asset_id", "unknown")),
            embedding=latent_snapshot.embedding,
            escalating=escalating,
        )

        relationship_names = [f"{str(r.get('source',''))}->{str(r.get('target',''))}" for r in list(observation.get("top_relationships") or [])[:3]]
        traj = self.trajectory_memory.update(
            asset_id=str(observation.get("asset_id", "unknown")),
            embedding=latent_snapshot.embedding,
            escalating=escalating,
            in_critical_region=in_critical,
            phase_shift=transition.transition_path in {"escalating", "reversible"},
            mechanism_names=relationship_names,
        )
        mech = self.mechanisms.update(
            top_relationships=list(observation.get("top_relationships") or []),
            subsystem_impact=dict(observation.get("subsystem_impact") or {}),
            escalating=escalating,
            trajectory_family=str(traj.get("current_trajectory_path_family", transition.transition_path)),
            recovering=recovering,
        )
        forecast = self.trajectory_forecast.forecast(
            trajectory_intelligence=traj,
            transition_dynamics={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
            },
        )
        laws = self.law_extractor.update(
            trajectory_info=traj,
            mechanism_info=mech,
            transition={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
            },
            counterfactual_info=cf,
            asset_id=str(observation.get("asset_id", "unknown")),
        )
        baseline_risk_assessment = {
            "current_risk_level": "high"
            if float(transition.escalation_probability) >= 0.7
            else ("medium" if float(transition.escalation_probability) >= 0.45 else "low"),
            "projected_score": float(transition.escalation_probability),
        }
        baseline_operator_guidance = {
            "recommended_actions": [
                "Inspect subsystem/cluster first.",
                "Validate top linked sensor pair and calibration.",
            ]
        }
        law_decision_support = self.law_engine.evaluate(
            law_candidates=laws,
            trajectory_info=traj,
            mechanism_info=mech,
            risk_assessment=baseline_risk_assessment,
            operator_guidance=baseline_operator_guidance,
            counterfactuals=cf,
        )

        intervention_intelligence = self.intervention_intelligence.update(
            asset_id=str(observation.get("asset_id", "unknown")),
            observation={**observation, "latent_embedding": latent_snapshot.embedding},
            transition={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
                "reversibility_score": transition.reversibility_score,
                "distance_to_critical_region": transition.distance_to_critical_region,
            },
            trajectory=traj,
            mechanism=mech,
            laws=laws,
            counterfactuals=cf,
        )

        output = {
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
            "counterfactuals": cf,
            "archetype_intelligence": arch,
            "trajectory_archetypes": traj,
            "trajectory_forecast": forecast,
            "structural_law_candidates": laws,
            "law_engine_decision": law_decision_support,
            "mechanism_discovery": mech,
            "intervention_intelligence": intervention_intelligence,
        }
        output["compatibility"] = to_operator_compatibility(output)
        return output
