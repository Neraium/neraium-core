from __future__ import annotations

from typing import Any

from .simulation import StructuralTransitionSimulator


class StructuralBranchingEngine:
    def __init__(self, simulator: StructuralTransitionSimulator) -> None:
        self.simulator = simulator

    def build_branches(
        self,
        *,
        initial_embedding: list[float],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        laws: dict[str, Any],
        intervention: dict[str, Any],
        reliability: dict[str, Any],
    ) -> dict[str, Any]:
        top_law = ((laws.get("law_candidates") or [{}])[:1] or [{}])[0]
        best = (((intervention.get("recommendation") or {}).get("best_intervention") or {}))
        reduction = float(best.get("predicted_escalation_reduction", 0.12))

        branch_specs = [
            ("no_intervention", "baseline", None, 0.0),
            ("best_ranked_intervention", "intervention_conditioned", {"expected_escalation_reduction": reduction}, 0.0),
            ("exploratory_intervention", "intervention_conditioned", {"expected_escalation_reduction": reduction * 0.7}, -0.005),
            ("adverse_case", "adversarial", None, 0.03),
        ]
        branches: list[dict[str, Any]] = []
        for name, mode, intervention_cfg, perturbation in branch_specs:
            sim = self.simulator.simulate(
                initial_embedding=initial_embedding,
                transition=transition,
                trajectory=trajectory,
                law_candidate=top_law,
                intervention=intervention_cfg,
                reliability=reliability,
                mode=mode,
                perturbation=perturbation,
            )
            final = (sim.get("future_states") or [{}])[-1]
            risk_trace = sim.get("risk_evolution") or []
            branches.append(
                {
                    "branch": name,
                    "mode": mode,
                    "projected_trajectory_family": sim.get("trajectory_family_projection"),
                    "future_states": sim.get("future_states", []),
                    "escalation_probability": final.get("escalation_probability", 0.0),
                    "reversibility": final.get("reversibility_score", 0.0),
                    "uncertainty_band": {
                        "low": round(max(0.0, float(final.get("escalation_probability", 0.0)) - 0.12), 6),
                        "high": round(min(1.0, float(final.get("escalation_probability", 0.0)) + 0.12), 6),
                    },
                    "divergence_point_step": self._divergence_point(risk_trace),
                }
            )

        comp = self._compare(branches)
        return {
            "scenario_branches": branches,
            "branch_comparison": comp,
            "dominant_divergence_factors": comp.get("dominant_divergence_factors", []),
        }

    def _divergence_point(self, risk_trace: list[float]) -> int | None:
        for i in range(1, len(risk_trace)):
            if abs(float(risk_trace[i]) - float(risk_trace[0])) >= 0.12:
                return i + 1
        return None

    def _compare(self, branches: list[dict[str, Any]]) -> dict[str, Any]:
        if not branches:
            return {"status": "empty"}
        by_name = {b["branch"]: b for b in branches}
        baseline = by_name.get("no_intervention", branches[0])
        base_risk = float(baseline.get("escalation_probability", 0.0))
        deltas = {
            b["branch"]: round(float(b.get("escalation_probability", 0.0)) - base_risk, 6)
            for b in branches
        }
        sorted_risk = sorted(branches, key=lambda b: float(b.get("escalation_probability", 0.0)))
        dominant = []
        if deltas.get("adverse_case", 0.0) > 0.08:
            dominant.append("failure_to_intervene")
        if deltas.get("best_ranked_intervention", 0.0) < -0.04:
            dominant.append("intervention_effectiveness")
        if max(deltas.values()) - min(deltas.values()) > 0.14:
            dominant.append("law_sensitive_dynamics")

        return {
            "baseline_branch": baseline["branch"],
            "risk_delta_vs_baseline": deltas,
            "lowest_risk_branch": sorted_risk[0]["branch"],
            "highest_risk_branch": sorted_risk[-1]["branch"],
            "dominant_divergence_factors": dominant,
        }
