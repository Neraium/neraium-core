from __future__ import annotations

from typing import Any

from ..experiment_design.confusion import detect_structural_confusion
from ..experiment_design.discriminative import identify_discriminative_experiments
from ..experiment_design.exploration import score_exploration_vs_exploitation
from ..experiment_design.planner import build_experiment_plan
from ..hypothesis.tracker import track_competing_hypotheses
from ..observability.recommendation import recommend_measurements
from .efficiency import LearningEfficiencyTracker
from .uncertainty import decompose_uncertainty


class ActiveStructuralLearningEngine:
    """Experimental advisory layer for active experiment and intervention design."""

    def __init__(self) -> None:
        self.efficiency = LearningEfficiencyTracker(window=30)

    def update(
        self,
        *,
        observation: dict[str, Any],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        mechanism: dict[str, Any],
        laws: dict[str, Any],
        intervention: dict[str, Any],
        reliability: dict[str, Any],
        cross_system: dict[str, Any],
    ) -> dict[str, Any]:
        hypothesis_state = track_competing_hypotheses(
            trajectory=trajectory,
            mechanism=mechanism,
            laws=laws,
            transition=transition,
        )
        hypotheses = list(hypothesis_state.get("active_hypotheses") or [])

        uncertainty = decompose_uncertainty(
            transition=transition,
            trajectory=trajectory,
            hypotheses=hypotheses,
            reliability=reliability,
            cross_system=cross_system,
        )

        intervention_ranked = (intervention.get("recommendation") or {}).get("ranked_interventions") or []
        experiments = identify_discriminative_experiments(
            hypotheses=hypotheses,
            uncertainty=uncertainty,
            intervention_ranking=intervention_ranked,
        )

        exploration = score_exploration_vs_exploitation(
            intervention=intervention,
            candidate_experiments=list(experiments.get("candidate_experiments") or []),
        )

        measurements = recommend_measurements(
            observation=observation,
            trajectory=trajectory,
            hypotheses=hypotheses,
            uncertainty=uncertainty,
        )

        plan = build_experiment_plan(
            hypotheses=hypotheses,
            candidate_experiments=list(experiments.get("candidate_experiments") or []),
            measurements=list(measurements.get("recommended_measurements") or []),
            uncertainty=uncertainty,
        )

        confusion = detect_structural_confusion(
            hypotheses=hypotheses,
            trajectory=trajectory,
            reliability=reliability,
        )

        efficiency = self.efficiency.update(
            uncertainty_magnitude=float(uncertainty.get("uncertainty_magnitude", 0.0)),
            top_hypothesis_likelihood=float(hypotheses[0].get("likelihood", 0.0) if hypotheses else 0.0),
            info_gain=float(experiments.get("expected_information_gain", 0.0)),
        )

        return {
            "status": "experimental",
            "advisory_only": True,
            "safety_boundary": {
                "auto_intervention_enabled": False,
                "overrides_existing_recommendation_layer": False,
            },
            "uncertainty": uncertainty,
            "hypotheses": hypotheses,
            "hypothesis_confidence_distribution": hypothesis_state.get("hypothesis_confidence_distribution") or [],
            "candidate_experiments": experiments.get("candidate_experiments") or [],
            "expected_information_gain": experiments.get("expected_information_gain", 0.0),
            "discriminative_power": experiments.get("discriminative_power", 0.0),
            "exploration_vs_exploitation": exploration,
            "recommended_measurements": measurements.get("recommended_measurements") or [],
            "observability_gap_score": measurements.get("observability_gap_score", 0.0),
            "measurement_uncertainty_reduction": measurements.get("expected_uncertainty_reduction", 0.0),
            "experiment_plan": plan.get("experiment_plan") or {},
            "confusion_flags": [confusion],
            "learning_efficiency": efficiency,
        }
