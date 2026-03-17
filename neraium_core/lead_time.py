from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class LeadTimeEstimator:
    failure_threshold: float = 5.0
    min_velocity: float = 1e-6
    max_horizon: float = 240.0

    def estimate_failure_time(self, drift_history: Iterable[float]):
        history = [float(value) for value in drift_history]
        if len(history) < 2:
            return None

        current = history[-1]
        velocity = history[-1] - history[-2]
        if velocity <= self.min_velocity:
            return None

        hours = (self.failure_threshold - current) / velocity
        if hours < 0:
            return 0.0

        return min(float(hours), self.max_horizon)
