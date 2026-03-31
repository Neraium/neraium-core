from __future__ import annotations

from typing import Any

from ..active_learning import ActiveStructuralLearningEngine
from ..falsification import StructuralFalsificationEngine
from ..federation.layer import CrossSystemStructuralIntelligenceLayer
from ..law_engine import StructuralLawDecisionEngine
from ..law_extraction.extractor import StructuralLawExtractor
from ..mechanisms.discovery import MechanismDiscoveryLayer
from ..sandbox import StructuralSandboxEngine
from ..universal import ExperimentalUniversalStructuralLayer


class ResearchExperimentalOrchestrator:
    """Advisory and experimental intelligence paths that cannot directly drive production recommendations."""

    def __init__(self) -> None:
        self.mechanisms = MechanismDiscoveryLayer()
        self.law_extractor = StructuralLawExtractor()
        self.law_engine = StructuralLawDecisionEngine()
        self.cross_system = CrossSystemStructuralIntelligenceLayer()
        self.experimental_universal = ExperimentalUniversalStructuralLayer()
        self.falsification = StructuralFalsificationEngine()
        self.sandbox = StructuralSandboxEngine()
        self.active_learning = ActiveStructuralLearningEngine()

    def update(
        self,
        *,
        observation: dict[str, Any],
        production: dict[str, Any],
        include_experimental: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        transition = production.get("transition_dynamics") or {}
        trajectory = production.get("trajectory_archetypes") or {}
        counterfactuals = production.get("counterfactuals") or {}
        intervention = production.get("intervention_intelligence") or {}
        reliability = production.get("reliability_intelligence") or {}

        escalating = str(transition.get("transition_path", "")) == "escalating" or float(transition.get("escalation_probability", 0.0)) > 0.65
        recovering = str(transition.get("transition_path", "")) == "reversible"

        mechanism = self.mechanisms.update(
            top_relationships=list(observation.get("top_relationships") or []),
            subsystem_impact=dict(observation.get("subsystem_impact") or {}),
            escalating=escalating,
            trajectory_family=str(trajectory.get("current_trajectory_path_family", transition.get("transition_path", "unknown"))),
            recovering=recovering,
        )
        laws = self.law_extractor.update(
            trajectory_info=trajectory,
            mechanism_info=mechanism,
            transition={
                "regime": transition.get("regime", "unknown"),
                "transition_path": transition.get("transition_path", "unknown"),
                "escalation_probability": float(transition.get("escalation_probability", 0.0)),
            },
            counterfactual_info=counterfactuals,
            asset_id=str(observation.get("asset_id", "unknown")),
        )
        baseline_risk_assessment = {
            "current_risk_level": "high"
            if float(transition.get("escalation_probability", 0.0)) >= 0.7
            else ("medium" if float(transition.get("escalation_probability", 0.0)) >= 0.45 else "low"),
            "projected_score": float(transition.get("escalation_probability", 0.0)),
        }
        law_decision_support = self.law_engine.evaluate(
            law_candidates=laws,
            trajectory_info=trajectory,
            mechanism_info=mechanism,
            risk_assessment=baseline_risk_assessment,
            operator_guidance={"recommended_actions": ["Inspect subsystem/cluster first."]},
            counterfactuals=counterfactuals,
        )
        cross_system = self.cross_system.update(
            trajectory_info=trajectory,
            law_info=laws,
            intervention_info=intervention,
        )

        advisory = {
            "mechanism_discovery": mechanism,
            "structural_law_candidates": laws,
            "law_engine_decision": law_decision_support,
            **cross_system,
        }

        if not include_experimental:
            return advisory, {}

        top_mech = str((((mechanism.get("mechanism_candidates") or [{}])[0]).get("mechanism", "unknown")))
        universal = self.experimental_universal.update(
            system_id=str(observation.get("asset_id", "unknown")),
            system_type=str(observation.get("system_type") or observation.get("asset_type") or "unknown"),
            domain=str(observation.get("domain") or observation.get("industry") or "unknown"),
            latent_trajectory=(production.get("latent_structural_state") or {}).get("trajectory", []),
            escalating=escalating,
            mechanism_name=top_mech,
            intervention_info=intervention,
        )
        falsification = self.falsification.update(
            domain=str(observation.get("domain") or observation.get("industry") or "unknown"),
            transition={
                "regime": transition.get("regime", "unknown"),
                "transition_path": transition.get("transition_path", "unknown"),
                "escalation_probability": float(transition.get("escalation_probability", 0.0)),
            },
            trajectory=trajectory,
            laws=laws,
            intervention=intervention,
            universal=universal,
        )
        active_learning = self.active_learning.update(
            observation=observation,
            transition={
                "regime": transition.get("regime", "unknown"),
                "transition_path": transition.get("transition_path", "unknown"),
                "escalation_probability": float(transition.get("escalation_probability", 0.0)),
                "uncertainty": float(transition.get("uncertainty", 0.0)),
            },
            trajectory=trajectory,
            mechanism=mechanism,
            laws=laws,
            intervention=intervention,
            reliability=reliability,
            cross_system=(cross_system.get("cross_system_structural_intelligence") or {}),
        )
        sandbox = self.sandbox.evaluate(
            latent_state=production.get("latent_structural_state") or {},
            transition=transition,
            trajectory=trajectory,
            laws=laws,
            intervention=intervention,
            reliability=reliability,
        )

        experimental = {
            **universal,
            **falsification,
            "active_learning": active_learning,
            **sandbox,
        }
        return advisory, experimental
