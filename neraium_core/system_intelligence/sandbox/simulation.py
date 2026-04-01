from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    horizon: int = 8
    step_size: float = 0.12


class StructuralTransitionSimulator:
    """Bounded latent-state simulator for structural sandboxing."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    @staticmethod
    def _regime(radius: float) -> str:
        if radius < 0.75:
            return "stable"
        if radius < 1.35:
            return "transitional"
        return "critical"

    @staticmethod
    def _path_from_delta(delta_radius: float) -> str:
        if abs(delta_radius) < 0.015:
            return "stable"
        if delta_radius > 0.04:
            return "escalating"
        if delta_radius < -0.03:
            return "reversible"
        return "drifting"

    def _mode_multiplier(self, mode: str) -> float:
        return {
            "baseline": 1.0,
            "law_conditioned": 1.05,
            "archetype_conditioned": 1.0,
            "intervention_conditioned": 0.82,
            "adversarial": 1.28,
        }.get(mode, 1.0)

    def simulate(
        self,
        *,
        initial_embedding: list[float],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        law_candidate: dict[str, Any] | None = None,
        intervention: dict[str, Any] | None = None,
        reliability: dict[str, Any] | None = None,
        mode: str = "baseline",
        horizon: int | None = None,
        perturbation: float = 0.0,
    ) -> dict[str, Any]:
        steps = int(horizon or self.config.horizon)
        z = np.asarray(initial_embedding, dtype=float)
        radius = float(np.linalg.norm(z))
        if radius <= 1e-8:
            z = np.array([0.1, 0.1, 0.1], dtype=float)
            radius = float(np.linalg.norm(z))

        escal = float(transition.get("escalation_probability", 0.0))
        rev = float(transition.get("reversibility_score", 0.5))
        novelty = float(trajectory.get("novelty_score", 0.5))

        law_effect = float((law_candidate or {}).get("estimated_effect", 0.0))
        law_support = float((law_candidate or {}).get("confidence", 0.0))
        intervention_shift = float((intervention or {}).get("expected_escalation_reduction", 0.0))
        reliability_conf = float(((reliability or {}).get("simulation_confidence") or 0.6))

        regime_changes: list[dict[str, Any]] = []
        states: list[dict[str, Any]] = []
        risk_trace: list[float] = []

        mode_mult = self._mode_multiplier(mode)
        prev_regime = self._regime(radius)

        for step in range(1, steps + 1):
            directional = (escal - 0.5) * 0.18 + (0.25 - rev) * 0.06
            directional += law_effect * (0.04 + 0.06 * law_support)
            directional -= intervention_shift * 0.12
            directional *= mode_mult
            if mode == "archetype_conditioned":
                fam = str(trajectory.get("current_trajectory_path_family", "drift"))
                if fam in {"oscillatory", "reversible"}:
                    directional += 0.05 * np.sin(step)
                if fam in {"rapid-collapse", "escalating"}:
                    directional += 0.02
            directional += float(perturbation)
            directional *= 0.7 + 0.3 * (1.0 - novelty)

            unit = z / (np.linalg.norm(z) + 1e-8)
            tangential = np.roll(unit, 1) * 0.03 * np.sin(step / 2.0)
            dz = unit * directional + tangential
            z = z + self.config.step_size * dz

            new_radius = float(np.linalg.norm(z))
            delta_radius = new_radius - radius
            radius = new_radius

            regime = self._regime(radius)
            if regime != prev_regime:
                regime_changes.append({"step": step, "from": prev_regime, "to": regime})
            prev_regime = regime

            risk = max(0.0, min(1.0, escal + 0.3 * (radius - 0.8)))
            risk_trace.append(round(float(risk), 6))

            sim_uncertainty = max(0.06, min(0.92, (1.0 - reliability_conf) + 0.25 * novelty + 0.02 * step))
            states.append(
                {
                    "step": step,
                    "embedding": [round(float(v), 6) for v in z.tolist()],
                    "regime": regime,
                    "transition_path": self._path_from_delta(delta_radius),
                    "escalation_probability": round(float(risk), 6),
                    "reversibility_score": round(float(max(0.0, min(1.0, rev + intervention_shift * 0.3 - novelty * 0.2))), 6),
                    "distance_to_critical_region": round(float(max(0.0, 1.35 - radius)), 6),
                    "uncertainty": round(float(sim_uncertainty), 6),
                }
            )

        return {
            "mode": mode,
            "future_states": states,
            "regime_changes": regime_changes,
            "risk_evolution": risk_trace,
            "trajectory_family_projection": str(trajectory.get("current_trajectory_path_family", "drift")),
        }
