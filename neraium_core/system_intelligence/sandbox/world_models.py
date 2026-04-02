from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .simulation import StructuralTransitionSimulator


@dataclass(frozen=True)
class WorldModelHypothesis:
    model_id: str
    description: str
    mode: str
    perturbation: float


class StructuralWorldModelEngine:
    def __init__(self, simulator: StructuralTransitionSimulator) -> None:
        self.simulator = simulator

    def default_models(self) -> list[WorldModelHypothesis]:
        return [
            WorldModelHypothesis(
                model_id="drift_dominated_degradation",
                description="Assumes monotonic latent drift toward critical region.",
                mode="law_conditioned",
                perturbation=0.02,
            ),
            WorldModelHypothesis(
                model_id="oscillatory_instability",
                description="Assumes oscillatory fluctuations with intermittent escalation bursts.",
                mode="archetype_conditioned",
                perturbation=0.0,
            ),
            WorldModelHypothesis(
                model_id="localized_failure",
                description="Assumes localized subsystem failure causing abrupt escalation pressure.",
                mode="adversarial",
                perturbation=0.035,
            ),
        ]

    def compare(
        self,
        *,
        initial_embedding: list[float],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        laws: dict[str, Any],
        reliability: dict[str, Any],
        observed_future: list[list[float]] | None,
    ) -> dict[str, Any]:
        top_law = ((laws.get("law_candidates") or [{}])[:1] or [{}])[0]
        models = self.default_models()
        outputs = []

        for wm in models:
            sim = self.simulator.simulate(
                initial_embedding=initial_embedding,
                transition=transition,
                trajectory=trajectory,
                law_candidate=top_law,
                reliability=reliability,
                mode=wm.mode,
                perturbation=wm.perturbation,
            )
            fit = self._fit_score(sim.get("future_states", []), observed_future)
            outputs.append(
                {
                    "model_id": wm.model_id,
                    "description": wm.description,
                    "projection": sim,
                    "model_fit": round(float(fit), 6),
                    "discriminative_features": self._discriminative_features(sim.get("future_states", [])),
                }
            )

        outputs.sort(key=lambda row: row["model_fit"], reverse=True)
        best = outputs[0] if outputs else None
        confidence_gap = 0.0
        if len(outputs) > 1:
            confidence_gap = float(outputs[0]["model_fit"] - outputs[1]["model_fit"])
        return {
            "competing_world_models": outputs,
            "world_model_fit_scores": {item["model_id"]: item["model_fit"] for item in outputs},
            "best_explaining_model": best["model_id"] if best else None,
            "discriminating_observations": best["discriminative_features"] if best else [],
            "confidence_gap": round(float(max(0.0, confidence_gap)), 6),
        }

    def _fit_score(self, predicted_states: list[dict[str, Any]], observed_future: list[list[float]] | None) -> float:
        if not observed_future:
            return 0.45
        pred = np.asarray([s["embedding"] for s in predicted_states[: len(observed_future)]], dtype=float)
        obs = np.asarray(observed_future[: len(pred)], dtype=float)
        if pred.size == 0 or obs.size == 0:
            return 0.45
        err = float(np.mean(np.linalg.norm(pred - obs, axis=1)))
        return max(0.0, min(1.0, 1.0 - err / 2.5))

    def _discriminative_features(self, predicted_states: list[dict[str, Any]]) -> list[str]:
        if not predicted_states:
            return []
        risks = [float(x.get("escalation_probability", 0.0)) for x in predicted_states]
        spread = max(risks) - min(risks)
        features = [f"risk_spread={spread:.3f}"]
        regimes = [str(x.get("regime", "unknown")) for x in predicted_states]
        if "critical" in regimes:
            features.append("critical_reach_detected")
        if any(str(x.get("transition_path")) == "reversible" for x in predicted_states):
            features.append("recovery_signal_present")
        return features
