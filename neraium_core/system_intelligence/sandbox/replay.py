from __future__ import annotations

from typing import Any

import numpy as np

from .divergence import StructuralDivergenceAnalyzer
from .simulation import StructuralTransitionSimulator


class StructuralReplayEngine:
    def __init__(self, simulator: StructuralTransitionSimulator) -> None:
        self.simulator = simulator
        self.divergence = StructuralDivergenceAnalyzer()

    def replay(
        self,
        *,
        observed_trajectory: list[list[float]],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        laws: dict[str, Any],
        intervention: dict[str, Any],
        reliability: dict[str, Any],
    ) -> dict[str, Any]:
        if len(observed_trajectory) < 3:
            return {
                "replay_trace": [],
                "prediction_error_trace": [],
                "law_support_trace": [],
                "intervention_effect_trace": [],
                "divergence_analysis": {
                    "divergence_type": "insufficient_data",
                    "divergence_magnitude": 0.0,
                    "corrective_actions": ["collect_more_history"],
                },
            }

        top_law = ((laws.get("law_candidates") or [{}])[:1] or [{}])[0]
        law_conf = float(top_law.get("confidence", 0.0))
        intervention_support = float((((intervention.get("recommendation") or {}).get("best_intervention") or {}).get("confidence", 0.0)))
        replay_trace = []
        error_trace = []
        law_trace = []
        intervention_trace = []

        for idx in range(1, len(observed_trajectory) - 1):
            hist = observed_trajectory[: idx + 1]
            pred = self.simulator.simulate(
                initial_embedding=hist[-1],
                transition=transition,
                trajectory=trajectory,
                law_candidate=top_law,
                reliability=reliability,
                mode="law_conditioned",
                horizon=1,
            )
            predicted_next = (pred.get("future_states") or [{}])[0].get("embedding") or hist[-1]
            observed_next = observed_trajectory[idx + 1]
            error = float(np.linalg.norm(np.asarray(predicted_next, dtype=float) - np.asarray(observed_next, dtype=float)))
            error_trace.append(round(error, 6))
            replay_trace.append(
                {
                    "step": idx,
                    "predicted_next": [round(float(v), 6) for v in predicted_next],
                    "observed_next": [round(float(v), 6) for v in observed_next],
                    "is_accurate": error <= 0.2,
                }
            )
            law_trace.append({"step": idx, "law_id": top_law.get("law_id"), "support": round(law_conf, 6)})
            intervention_trace.append(
                {
                    "step": idx,
                    "simulated_effect": round(float(max(0.0, min(1.0, 0.2 + 0.5 * intervention_support - error * 0.3))), 6),
                    "historical_evidence_support": round(intervention_support, 6),
                    "recommended_action": ((intervention.get("recommendation") or {}).get("best_intervention") or {}).get("name"),
                }
            )

        divergence = self.divergence.analyze(
            prediction_errors=error_trace,
            trajectory=trajectory,
            law_support=law_conf,
            intervention_support=intervention_support,
        )
        return {
            "replay_trace": replay_trace,
            "prediction_error_trace": error_trace,
            "law_support_trace": law_trace,
            "intervention_effect_trace": intervention_trace,
            "divergence_analysis": divergence,
        }
