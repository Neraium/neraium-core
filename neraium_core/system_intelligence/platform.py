from __future__ import annotations

from typing import Any

from .adapters.compatibility import to_operator_compatibility
from .archetypes.archetype_memory import StructuralArchetypeMemory
from .counterfactuals.intervention_engine import CounterfactualInterventionEngine
from .forecast.trajectory_conditioned import TrajectoryConditionedForecaster
from .law_extraction.extractor import StructuralLawExtractor
from .mechanisms.discovery import MechanismDiscoveryLayer
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
            "mechanism_discovery": mech,
        }
        output["compatibility"] = to_operator_compatibility(output)
        return output
