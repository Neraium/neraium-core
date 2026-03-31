from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransitionAssessment:
    regime: str
    transition_path: str
    escalation_probability: float
    reversibility_score: float
    distance_to_critical_region: float
    uncertainty: float


class LatentTransitionModel:
    """Simple, inspectable transition model over latent state trajectories."""

    def __init__(self, critical_radius: float = 1.35) -> None:
        self.critical_radius = float(critical_radius)
        self._history: deque[np.ndarray] = deque(maxlen=180)
        self._transition_counts: dict[tuple[str, str], int] = defaultdict(int)
        self._state_counts: dict[str, int] = defaultdict(int)
        self._last_regime: str | None = None

    @staticmethod
    def _radius_regime(radius: float) -> str:
        if radius < 0.75:
            return "stable"
        if radius < 1.35:
            return "transitional"
        return "critical"

    def assess(self, embedding: list[float]) -> TransitionAssessment:
        z = np.asarray(embedding, dtype=float)
        self._history.append(z)
        radius = float(np.linalg.norm(z))
        regime = self._radius_regime(radius)

        if self._last_regime is not None:
            self._transition_counts[(self._last_regime, regime)] += 1
        self._state_counts[regime] += 1
        self._last_regime = regime

        if len(self._history) >= 2:
            diff = self._history[-1] - self._history[-2]
            radial_derivative = float(np.dot(diff, self._history[-1]) / (np.linalg.norm(self._history[-1]) + 1e-6))
        else:
            radial_derivative = 0.0

        if abs(radial_derivative) < 0.01:
            path = "stable"
        elif radial_derivative > 0.06:
            path = "escalating"
        elif radial_derivative < -0.04:
            path = "reversible"
        else:
            path = "drifting"

        total_from_regime = sum(v for (a, _), v in self._transition_counts.items() if a == regime)
        toward_critical = sum(v for (a, b), v in self._transition_counts.items() if a == regime and b == "critical")
        empirical = toward_critical / total_from_regime if total_from_regime > 0 else 0.0
        radial_prior = max(0.0, min(1.0, (radius - 0.8) / 1.2))
        escalation_probability = float(max(0.0, min(1.0, 0.55 * empirical + 0.45 * radial_prior)))

        backward = self._transition_counts.get(("critical", "transitional"), 0) + self._transition_counts.get(("transitional", "stable"), 0)
        forward = self._transition_counts.get(("stable", "transitional"), 0) + self._transition_counts.get(("transitional", "critical"), 0)
        reversibility = float((backward + 1.0) / (backward + forward + 2.0))

        distance_to_critical = float(max(0.0, self.critical_radius - radius))
        uncertainty = float(max(0.05, min(0.9, 1.0 / np.sqrt(max(2, len(self._history))))))

        return TransitionAssessment(
            regime=regime,
            transition_path=path,
            escalation_probability=escalation_probability,
            reversibility_score=reversibility,
            distance_to_critical_region=distance_to_critical,
            uncertainty=uncertainty,
        )
