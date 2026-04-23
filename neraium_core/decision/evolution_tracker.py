"""Evolution tracker: Track system progression through regime space over time.

Maintains evolution history, detects regime transitions, and classifies
degradation patterns. This is the foundation for evolution-aware decisioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from neraium_core.decision.models import SeverityLevel


@dataclass
class RegimeState:
    """Snapshot of system state at a point in time."""

    regime: str  # "STABLE", "WATCH", "ALERT", "CRITICAL"
    frame_index: int
    severity: SeverityLevel
    drift_score: float
    relational_instability: float
    regime_distance: float  # Distance to nearest severity threshold
    confidence: float  # Confidence in this regime assignment [0, 1]


@dataclass
class EvolutionTrajectory:
    """Evolution context: how system has moved through regime space."""

    regime_sequence: list[str] = field(default_factory=list)
    transition_points: list[int] = field(default_factory=list)  # Frame indices
    current_regime_persistence: int = 0  # Frames in current regime
    regime_distances: list[float] = field(default_factory=list)  # Distance evolution
    is_degrading: bool = False  # Overall direction: toward higher severity?
    movement_pattern: str = "stable"  # "linear", "oscillating", "accelerating"
    regime_entry_frame: int = 0  # When did system enter current regime?
    regime_transitions_count: int = 0  # Total transitions seen


def _regime_from_severity(severity: SeverityLevel) -> str:
    """Map severity level to regime name."""
    mapping = {
        "LOW": "STABLE",
        "MODERATE": "WATCH",
        "ELEVATED": "ALERT",
        "HIGH": "CRITICAL",
    }
    return mapping.get(severity, "STABLE")


class EvolutionTracker:
    """Tracks regime progression and detects structural evolution patterns."""

    HISTORY_WINDOW = 30  # Keep last 30 frames of history
    MIN_TRANSITION_FRAMES = 2  # Minimum frames to confirm regime change

    def __init__(self):
        self.history: list[RegimeState] = []
        self.trajectory = EvolutionTrajectory()
        self.previous_regime: Optional[str] = None
        self.frame_counter = 0

    def update_frame(
        self,
        severity: SeverityLevel,
        drift_score: float,
        relational_instability: float,
        regime_distance: float,
        finding_confidence: float,
    ) -> EvolutionTrajectory:
        """Update tracker with new frame data.

        Args:
            severity: Current severity classification
            drift_score: Structural drift [0, 1]
            relational_instability: Relational instability [0, 1]
            regime_distance: Distance to nearest threshold [0, 1]
            finding_confidence: Confidence in finding [0, 1]

        Returns:
            Updated EvolutionTrajectory
        """
        self.frame_counter += 1
        regime = _regime_from_severity(severity)

        # Create regime state snapshot
        state = RegimeState(
            regime=regime,
            frame_index=self.frame_counter,
            severity=severity,
            drift_score=drift_score,
            relational_instability=relational_instability,
            regime_distance=regime_distance,
            confidence=finding_confidence,
        )

        # Add to history (maintain window)
        self.history.append(state)
        if len(self.history) > self.HISTORY_WINDOW:
            self.history = self.history[-self.HISTORY_WINDOW :]

        # Update trajectory
        self._update_trajectory(regime)

        return self.trajectory

    def _update_trajectory(self, current_regime: str) -> None:
        """Update trajectory based on regime progression."""
        # Update regime sequence
        if not self.trajectory.regime_sequence or self.trajectory.regime_sequence[-1] != current_regime:
            self.trajectory.regime_sequence.append(current_regime)

        # Detect transitions
        if self.previous_regime and self.previous_regime != current_regime:
            # Confirm transition (avoid flip-flop)
            recent_regimes = [s.regime for s in self.history[-self.MIN_TRANSITION_FRAMES :]]
            if recent_regimes.count(current_regime) >= self.MIN_TRANSITION_FRAMES:
                if not self.trajectory.transition_points or self.trajectory.transition_points[-1] != self.frame_counter:
                    self.trajectory.transition_points.append(self.frame_counter)
                    self.trajectory.regime_transitions_count += 1
                    self.trajectory.regime_entry_frame = self.frame_counter

        # Update persistence
        frames_at_current = sum(1 for s in reversed(self.history) if s.regime == current_regime)
        self.trajectory.current_regime_persistence = frames_at_current

        # Update regime distance trajectory
        if self.history:
            self.trajectory.regime_distances.append(self.history[-1].regime_distance)
            if len(self.trajectory.regime_distances) > self.HISTORY_WINDOW:
                self.trajectory.regime_distances = self.trajectory.regime_distances[-self.HISTORY_WINDOW :]

        # Determine degradation direction
        self._update_degradation_direction()

        # Determine movement pattern
        self._update_movement_pattern()

        self.previous_regime = current_regime

    def _update_degradation_direction(self) -> None:
        """Determine if system is degrading toward higher severity."""
        if len(self.history) < 3:
            self.trajectory.is_degrading = False
            return

        severity_scores = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}
        recent_scores = [severity_scores.get(s.severity, 0) for s in self.history[-10 :]]

        if len(recent_scores) < 2:
            self.trajectory.is_degrading = False
            return

        # Simple trend: is latest higher than baseline?
        trend = recent_scores[-1] - recent_scores[0]
        self.trajectory.is_degrading = trend > 0.2

    def _update_movement_pattern(self) -> None:
        """Classify movement pattern: linear, oscillating, or accelerating."""
        if len(self.history) < 5:
            self.trajectory.movement_pattern = "stable"
            return

        # Extract drift scores from last 10 frames
        recent_drifts = [s.drift_score for s in self.history[-10 :]]

        if len(recent_drifts) < 3:
            self.trajectory.movement_pattern = "stable"
            return

        # Compute first and second derivatives to detect acceleration
        deltas = [recent_drifts[i + 1] - recent_drifts[i] for i in range(len(recent_drifts) - 1)]
        if not deltas:
            self.trajectory.movement_pattern = "stable"
            return

        # Check for acceleration (second derivative > 0)
        second_deltas = [deltas[i + 1] - deltas[i] for i in range(len(deltas) - 1)]
        mean_accel = sum(second_deltas) / len(second_deltas) if second_deltas else 0.0

        # Oscillation detection
        recent_history = self.history[-10:] if len(self.history) >= 2 else self.history
        regime_changes = sum(1 for i in range(1, len(recent_history)) if recent_history[i - 1].regime != recent_history[i].regime)

        if regime_changes >= 3:
            self.trajectory.movement_pattern = "oscillating"
        elif mean_accel > 0.05:
            self.trajectory.movement_pattern = "accelerating"
        else:
            self.trajectory.movement_pattern = "linear"

    def get_regime_entry_story(self) -> str:
        """Generate human-readable story of how system entered current regime."""
        if not self.trajectory.regime_sequence or not self.history:
            return "No regime history."

        current_regime = self.trajectory.regime_sequence[-1]
        frames_in_regime = self.trajectory.current_regime_persistence
        transitions = self.trajectory.regime_transitions_count

        if frames_in_regime == 1:
            return f"Just entered {current_regime}."

        if len(self.trajectory.regime_sequence) > 1:
            previous_regime = self.trajectory.regime_sequence[-2]
            return f"In {current_regime} for {frames_in_regime} frames (from {previous_regime}). Total transitions: {transitions}."

        return f"In {current_regime} for {frames_in_regime} frames."

    def get_trajectory_story(self) -> str:
        """Generate human-readable story of trajectory direction."""
        if self.trajectory.movement_pattern == "accelerating":
            direction = "accelerating degradation"
        elif self.trajectory.movement_pattern == "oscillating":
            direction = "oscillating instability"
        elif self.trajectory.is_degrading:
            direction = "degrading trajectory"
        else:
            direction = "stable progression"

        return f"Movement: {direction} ({self.trajectory.movement_pattern})"
