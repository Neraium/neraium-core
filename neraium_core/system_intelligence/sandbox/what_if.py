from __future__ import annotations

from typing import Any


class StructuralWhatIfEngine:
    def generate(
        self,
        *,
        baseline_branch: dict[str, Any],
        trajectory: dict[str, Any],
        transition: dict[str, Any],
    ) -> dict[str, Any]:
        baseline_risk = float(baseline_branch.get("escalation_probability", transition.get("escalation_probability", 0.0)))
        novelty = float(trajectory.get("novelty_score", 0.5))
        scenarios = [
            self._scenario("stabilize_subsystem_earlier", "Subsystem stabilization applied earlier in the trajectory.", baseline_risk, -0.08, novelty),
            self._scenario("damp_oscillation", "Oscillatory behavior damped before phase shift.", baseline_risk, -0.06, novelty),
            self._scenario("double_measurement_frequency", "Higher observability lowers epistemic uncertainty.", baseline_risk, -0.02, max(0.0, novelty - 0.1)),
            self._scenario("misclassify_as_drift", "Treat localized failure as drift-dominated behavior.", baseline_risk, 0.07, novelty + 0.12),
        ]
        return {
            "what_if_scenarios": scenarios,
            "scenario_delta_vs_baseline": {s["scenario_id"]: s["delta_vs_baseline"] for s in scenarios},
            "confidence_and_assumptions": {
                "bounded": True,
                "assumptions": [
                    "What-if outputs are exploratory and non-actionable by default.",
                    "Scenarios reuse existing structural representations and do not assume perfect actuation.",
                ],
            },
        }

    def _scenario(self, sid: str, description: str, base_risk: float, delta: float, novelty: float) -> dict[str, Any]:
        projected = max(0.0, min(1.0, base_risk + delta))
        confidence = max(0.05, min(0.9, 0.75 - novelty * 0.5 - abs(delta) * 0.8))
        return {
            "scenario_id": sid,
            "description": description,
            "projected_escalation_probability": round(projected, 6),
            "delta_vs_baseline": round(projected - base_risk, 6),
            "confidence": round(confidence, 6),
            "assumptions": ["bounded_structural_effect", "no_direct_operator_command"],
        }
