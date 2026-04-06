"""Alert state-machine for atomic layer detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class AlertTransition:
    timestamp: float | int
    level: str
    score: float
    dominant_detector: str


class AtomicAlertStateMachine:
    """Three-level GREEN/YELLOW/RED alerting state machine."""

    def __init__(self, green_yellow: float, yellow_red: float) -> None:
        self.green_yellow = float(green_yellow)
        self.yellow_red = float(yellow_red)
        self.level = "GREEN"
        self.history: List[AlertTransition] = []

    def update(self, score: float, dominant_detector: str, timestamp: float | int) -> str:
        next_level = self._level_for(score)
        if next_level != self.level:
            self.level = next_level
            self.history.append(
                AlertTransition(
                    timestamp=timestamp,
                    level=next_level,
                    score=float(score),
                    dominant_detector=dominant_detector,
                )
            )
        return self.level

    def _level_for(self, score: float) -> str:
        if score >= self.yellow_red:
            return "RED"
        if score >= self.green_yellow:
            return "YELLOW"
        return "GREEN"
