from __future__ import annotations

from typing import Any

from .adapters.compatibility import to_operator_compatibility
from .archetypes.archetype_memory import StructuralArchetypeMemory
from .counterfactuals.intervention_engine import CounterfactualInterventionEngine
from .federation.layer import CrossSystemStructuralIntelligenceLayer
from .forecast.trajectory_conditioned import TrajectoryConditionedForecaster
from .falsification import StructuralFalsificationEngine
from .intervention_intelligence.engine import InterventionIntelligenceEngine
from .law_engine import StructuralLawDecisionEngine
from .law_extraction.extractor import StructuralLawExtractor
from .mechanisms.discovery import MechanismDiscoveryLayer
from .reliability import StructuralReliabilityLayer
from .structural_state.latent_state import LatentStructuralStateEncoder
from .trajectory_memory.memory import CrossSystemTrajectoryMemory
from .transition_model.transition_dynamics import LatentTransitionModel
from .universal import ExperimentalUniversalStructuralLayer


class StructuralSystemIntelligencePlatform:
    """Integrated, bounded intelligence stack for structural state and intervention support."""

    def __init__(self) -> None:
        self.latent = LatentStructuralStateEncoder(latent_dim=3)
        self.transitions = LatentTransitionModel()
        self.counterfactuals = CounterfactualInterventionEngine()
        self.archetypes = StructuralArchetypeMemory()
        self.trajectory_memory = CrossSystemTrajectoryMemory(window_size=10)
        self.trajectory_forecast = TrajectoryConditionedForecaster()
        self.mechanisms = MechanismDiscoveryLayer()
        self.law_extractor = StructuralLawExtractor()
        self.law_engine = StructuralLawDecisionEngine()
        self.intervention_intelligence = InterventionIntelligenceEngine()
        self.cross_system = CrossSystemStructuralIntelligenceLayer()
        self.reliability = StructuralReliabilityLayer()
        self.experimental_universal = ExperimentalUniversalStructuralLayer()
        self.falsification = StructuralFalsificationEngine()

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

        relationship_names = [f"{str(r.get('source', ''))}->{str(r.get('target', ''))}" for r in list(observation.get("top_relationships") or [])[:3]]
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


        cross_system = self.cross_system.update(
            trajectory_info=traj,
            law_info=laws,
            intervention_info=intervention_intelligence,
        )
        self.reliability.finalize_due(
            asset_id=str(observation.get("asset_id", "unknown")),
            step=self.cross_system._step,
            transition={
                "escalation_probability": transition.escalation_probability,
                "transition_path": transition.transition_path,
                "regime": transition.regime,
            },
        )
        reliability = self.reliability.calibrate_all(
            asset_id=str(observation.get("asset_id", "unknown")),
            step=self.cross_system._step,
            transition={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
                "uncertainty": transition.uncertainty,
            },
            trajectory_forecast=forecast,
            laws=laws,
            intervention=intervention_intelligence,
            cross=(cross_system.get("cross_system_structural_intelligence") or {}),
        )
        rec_cal = float(((reliability.get("intervention_recommendation") or {}).get("recommendation_calibrated_confidence", 0.0)))
        if intervention_intelligence.get("recommendation") and intervention_intelligence["recommendation"].get("best_intervention"):
            intervention_intelligence["recommendation"]["best_intervention"]["raw_confidence"] = intervention_intelligence["recommendation"]["best_intervention"].get("confidence", 0.0)
            intervention_intelligence["recommendation"]["best_intervention"]["confidence"] = round(rec_cal, 6)

        top_mech = str((((mech.get("mechanism_candidates") or [{}])[0]).get("mechanism", "unknown")))
        universal = self.experimental_universal.update(
            system_id=str(observation.get("asset_id", "unknown")),
            system_type=str(observation.get("system_type") or observation.get("asset_type") or "unknown"),
            domain=str(observation.get("domain") or observation.get("industry") or "unknown"),
            latent_trajectory=latent_snapshot.trajectory,
            escalating=escalating,
            mechanism_name=top_mech,
            intervention_info=intervention_intelligence,
        )

        falsification = self.falsification.update(
            domain=str(observation.get("domain") or observation.get("industry") or "unknown"),
            transition={
                "regime": transition.regime,
                "transition_path": transition.transition_path,
                "escalation_probability": transition.escalation_probability,
            },
            trajectory=traj,
            laws=laws,
            intervention=intervention_intelligence,
            universal=universal,
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
            "reliability_intelligence": reliability,
            **cross_system,
            **universal,
            **falsification,
        }
        output["compatibility"] = to_operator_compatibility(output)
        return output
