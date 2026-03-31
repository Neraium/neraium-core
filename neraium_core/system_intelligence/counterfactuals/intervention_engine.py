from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CounterfactualScenario:
    name: str
    assumptions: list[str]
    projected_risk_score: float
    delta_vs_observed: float
    leverage_component: str


class CounterfactualInterventionEngine:
    """Approximate intervention engine with explicit assumptions, no causal guarantees."""

    def evaluate(self, observed: dict[str, Any], transition_escalation: float) -> dict[str, Any]:
        base = float(observed.get("composite_instability", 0.0))
        contributions = observed.get("contribution_scores") or {}
        top_component = "coupling"
        top_value = 0.0
        for key, value in contributions.items():
            v = float(value)
            if v > top_value:
                top_component, top_value = str(key), v

        scenarios = [
            CounterfactualScenario(
                name="remove_top_driver_contribution",
                assumptions=["driver contribution can be partially suppressed", "all other components held fixed"],
                projected_risk_score=max(0.0, base - 0.35 * top_value),
                delta_vs_observed=max(0.0, base - max(0.0, base - 0.35 * top_value)),
                leverage_component=top_component,
            ),
            CounterfactualScenario(
                name="restore_relationship_cluster_to_baseline",
                assumptions=["relationship cluster can be re-coupled", "graph deformation reduction propagates locally"],
                projected_risk_score=max(0.0, base - 0.20 * float(observed.get("graph_deformation_score", 0.0))),
                delta_vs_observed=max(0.0, base - max(0.0, base - 0.20 * float(observed.get("graph_deformation_score", 0.0)))),
                leverage_component="relationship_cluster",
            ),
            CounterfactualScenario(
                name="suppress_subsystem_instability",
                assumptions=["subsystem excitation can be damped", "regime transition responds proportionally"],
                projected_risk_score=max(0.0, base - 0.25 * float(observed.get("coupling_instability_score", 0.0))),
                delta_vs_observed=max(0.0, base - max(0.0, base - 0.25 * float(observed.get("coupling_instability_score", 0.0)))),
                leverage_component="subsystem",
            ),
        ]
        best = min(scenarios, key=lambda s: s.projected_risk_score)

        return {
            "status": "approximate",
            "observed_evidence": {
                "base_composite_instability": round(base, 4),
                "transition_escalation_probability": round(float(transition_escalation), 4),
            },
            "model_inference": {
                "highest_leverage_component": top_component,
                "assumption_boundary": "Counterfactuals are structural simulations under fixed-context assumptions; not formal causal effects.",
            },
            "scenario_rankings": [
                {
                    "name": s.name,
                    "projected_risk_score": round(float(s.projected_risk_score), 4),
                    "risk_delta": round(float(s.delta_vs_observed), 4),
                    "leverage_component": s.leverage_component,
                    "assumptions": s.assumptions,
                }
                for s in sorted(scenarios, key=lambda s: s.projected_risk_score)
            ],
            "best_intervention": {
                "name": best.name,
                "projected_risk_score": round(float(best.projected_risk_score), 4),
                "expected_escalation_reduction": round(float(max(0.0, transition_escalation * best.delta_vs_observed)), 4),
            },
        }
