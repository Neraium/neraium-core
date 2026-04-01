from __future__ import annotations

from collections import deque


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


class LearningEfficiencyTracker:
    def __init__(self, window: int = 20) -> None:
        self.uncertainty_history: deque[float] = deque(maxlen=max(3, window))
        self.top_likelihood_history: deque[float] = deque(maxlen=max(3, window))
        self.info_gain_history: deque[float] = deque(maxlen=max(3, window))

    def update(self, *, uncertainty_magnitude: float, top_hypothesis_likelihood: float, info_gain: float) -> dict[str, object]:
        self.uncertainty_history.append(float(uncertainty_magnitude))
        self.top_likelihood_history.append(float(top_hypothesis_likelihood))
        self.info_gain_history.append(float(info_gain))

        uncertainty_reduction = 0.0
        convergence_rate = 0.0
        if len(self.uncertainty_history) >= 2:
            uncertainty_reduction = self.uncertainty_history[0] - self.uncertainty_history[-1]
            convergence_rate = self.top_likelihood_history[-1] - self.top_likelihood_history[0]

        mean_gain = sum(self.info_gain_history) / max(1, len(self.info_gain_history))
        efficiency = _clip01(0.5 * _clip01(uncertainty_reduction + 0.5) + 0.3 * _clip01(convergence_rate + 0.5) + 0.2 * mean_gain)

        stagnation = []
        if len(self.uncertainty_history) >= 4 and uncertainty_reduction < 0.02:
            stagnation.append("uncertainty_not_reducing")
        if len(self.top_likelihood_history) >= 4 and convergence_rate < 0.02:
            stagnation.append("hypothesis_not_converging")

        return {
            "uncertainty_reduction_over_time": round(float(uncertainty_reduction), 4),
            "hypothesis_convergence_rate": round(float(convergence_rate), 4),
            "information_gain_per_intervention": round(float(mean_gain), 4),
            "learning_efficiency_score": round(float(efficiency), 4),
            "stagnation_flags": stagnation,
        }
