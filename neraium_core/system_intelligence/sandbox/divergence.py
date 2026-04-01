from __future__ import annotations

from typing import Any


class StructuralDivergenceAnalyzer:
    def analyze(
        self,
        *,
        prediction_errors: list[float],
        trajectory: dict[str, Any],
        law_support: float,
        intervention_support: float,
    ) -> dict[str, Any]:
        if not prediction_errors:
            return {
                "divergence_type": "insufficient_data",
                "divergence_magnitude": 0.0,
                "corrective_actions": ["increase_observation_history"],
            }

        mag = float(sum(prediction_errors) / len(prediction_errors))
        novelty = float(trajectory.get("novelty_score", 0.5))
        family_similarity = float(trajectory.get("family_similarity", 0.5))

        if novelty > 0.75:
            dtype = "novelty_unseen_behavior"
            actions = ["collect_new_archetype_examples", "reduce_actionability_until_recalibrated"]
        elif family_similarity < 0.45:
            dtype = "wrong_archetype_family"
            actions = ["recluster_trajectory_family", "expand_family_library"]
        elif law_support < 0.35:
            dtype = "wrong_law"
            actions = ["reweight_law_candidates", "run_targeted_falsification_experiments"]
        elif intervention_support < 0.3:
            dtype = "intervention_mismatch"
            actions = ["audit_intervention_preconditions", "separate_simulated_effect_from_evidence"]
        else:
            dtype = "insufficient_observability"
            actions = ["increase_measurement_frequency", "instrument_missing_subsystems"]

        return {
            "divergence_type": dtype,
            "divergence_magnitude": round(mag, 6),
            "corrective_actions": actions,
        }
