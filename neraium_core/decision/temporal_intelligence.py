"""Phase 3: Temporal Intelligence and Decision Coherence.

Implements temporal context tracking, trajectory-aware decisions, and consistency checking.
"""

from __future__ import annotations

from typing import Optional
from neraium_core.decision.models import (
    TemporalContext,
    SeverityLevel,
    TrajectoryLabel,
)


class TemporalIntelligence:
    """Tracks temporal patterns and enforces decision coherence."""

    WINDOW_SIZE = 10  # Track last 10 frames
    CONSISTENCY_WINDOW = 3  # Don't flip recommendations within 3 frames
    OSCILLATION_THRESHOLD = 2  # 2+ changes in 5 frames = oscillating

    def __init__(self):
        self.temporal_context: Optional[TemporalContext] = None

    def update_temporal_context(
        self,
        severity: SeverityLevel,
        drift_score: float,
        finding_confidence: float,
    ) -> TemporalContext:
        """Update temporal context with new frame data."""
        if self.temporal_context is None:
            self.temporal_context = TemporalContext()

        # Append to histories (keep last WINDOW_SIZE items)
        self.temporal_context.severity_history.append(severity)
        if len(self.temporal_context.severity_history) > self.WINDOW_SIZE:
            self.temporal_context.severity_history = (
                self.temporal_context.severity_history[-self.WINDOW_SIZE:]
            )

        self.temporal_context.drift_scores.append(drift_score)
        if len(self.temporal_context.drift_scores) > self.WINDOW_SIZE:
            self.temporal_context.drift_scores = (
                self.temporal_context.drift_scores[-self.WINDOW_SIZE:]
            )

        self.temporal_context.confidence_scores.append(finding_confidence)
        if len(self.temporal_context.confidence_scores) > self.WINDOW_SIZE:
            self.temporal_context.confidence_scores = (
                self.temporal_context.confidence_scores[-self.WINDOW_SIZE:]
            )

        # Update trajectory
        self.temporal_context.trajectory = self._compute_trajectory(
            self.temporal_context.severity_history
        )

        # Update drift velocity
        self.temporal_context.drift_velocity = self._compute_drift_velocity(
            self.temporal_context.drift_scores
        )

        # Detect oscillation
        self.temporal_context.oscillation_detected = self._detect_oscillation(
            self.temporal_context.severity_history
        )

        # Update confidence trend
        self.temporal_context.confidence_trend = self._compute_confidence_trend(
            self.temporal_context.confidence_scores
        )

        # Update persistent frames count
        self.temporal_context.persistent_frames_at_level = (
            self._count_persistent_frames(
                self.temporal_context.severity_history, severity
            )
        )

        return self.temporal_context

    def _compute_trajectory(self, severity_history: list[str]) -> TrajectoryLabel:
        """Compute trajectory: improving, stable, degrading, or unstable."""
        if len(severity_history) < 3:
            return "stable"

        severity_scores = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}

        recent = [severity_scores.get(s, 0) for s in severity_history[-5:]]
        if len(recent) < 2:
            return "stable"

        # Compute trend
        trend = (recent[-1] - recent[0]) / len(recent)

        # Oscillating?
        if self._detect_oscillation(severity_history):
            return "unstable"

        # Stable if minimal change
        if abs(trend) < 0.2:
            return "stable"

        # Degrading vs improving
        if trend > 0.2:
            return "degrading"
        else:
            return "improving"

    def _compute_drift_velocity(self, drift_scores: list[float]) -> float:
        """Compute rate of drift change: how fast is drift increasing/decreasing."""
        if len(drift_scores) < 2:
            return 0.0

        recent = drift_scores[-5:]
        if len(recent) < 2:
            return 0.0

        velocity = (recent[-1] - recent[0]) / len(recent)
        return max(-1.0, min(1.0, velocity))

    def _detect_oscillation(self, severity_history: list[str]) -> bool:
        """Detect if severity is oscillating between levels."""
        if len(severity_history) < 5:
            return False

        recent = severity_history[-5:]
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])

        return changes >= self.OSCILLATION_THRESHOLD

    def _compute_confidence_trend(self, confidence_scores: list[float]) -> float:
        """Compute confidence trend: is confidence increasing or decreasing."""
        if len(confidence_scores) < 2:
            return 0.0

        recent = confidence_scores[-5:]
        if len(recent) < 2:
            return 0.0

        trend = (recent[-1] - recent[0]) / len(recent)
        return max(-1.0, min(1.0, trend))

    def _count_persistent_frames(
        self, severity_history: list[str], current_severity: SeverityLevel
    ) -> int:
        """Count consecutive frames at current severity level."""
        count = 0
        for severity in reversed(severity_history):
            if severity == current_severity:
                count += 1
            else:
                break
        return count

    def get_trajectory_bias(self, trajectory: TrajectoryLabel) -> float:
        """Get decision bias based on trajectory.

        Returns:
            float: [-1, 1] where 1 = escalate, -1 = de-escalate, 0 = neutral
        """
        if trajectory == "degrading":
            return 0.3  # Slightly bias toward escalation
        elif trajectory == "improving":
            return -0.3  # Slightly bias toward caution
        elif trajectory == "unstable":
            return 0.2  # Moderate escalation for instability
        else:
            return 0.0  # Neutral for stable

    def should_accept_recommendation_flip(
        self,
        new_action: str,
        last_action: Optional[str],
        frames_since_last: int,
        severity_changed: bool,
    ) -> bool:
        """Check if we should allow a flip in recommendation direction.

        Within CONSISTENCY_WINDOW frames, only allow flip if severity changed.
        """
        if last_action is None or new_action == last_action:
            return True

        if frames_since_last >= self.CONSISTENCY_WINDOW:
            return True

        if severity_changed:
            return True

        return False

    def detect_state_transition(
        self,
        previous_trajectory: Optional[TrajectoryLabel],
        current_trajectory: TrajectoryLabel,
        previous_severity: Optional[SeverityLevel],
        current_severity: SeverityLevel,
    ) -> Optional[str]:
        """Detect meaningful state transitions.

        Returns:
            Transition event description if a real shift occurred, None otherwise.
        """
        if previous_trajectory is None or previous_severity is None:
            return None

        severity_levels = ["LOW", "MODERATE", "ELEVATED", "HIGH"]
        prev_idx = severity_levels.index(previous_severity) if previous_severity in severity_levels else 0
        curr_idx = severity_levels.index(current_severity) if current_severity in severity_levels else 0

        # Significant severity jump
        if curr_idx > prev_idx + 1:
            return f"severity_escalation:{previous_severity}→{current_severity}"

        # Major trajectory shift
        if previous_trajectory != current_trajectory:
            if previous_trajectory in {"stable", "improving"} and current_trajectory in {"degrading", "unstable"}:
                return f"trajectory_shift:stable→{current_trajectory}"
            elif previous_trajectory == "degrading" and current_trajectory == "unstable":
                return "trajectory_shift:degrading→unstable"

        return None

    def compute_confidence_delta(
        self, current_confidence: float
    ) -> float:
        """Compute change in confidence relative to recent history."""
        if len(self.temporal_context.confidence_scores) < 2 if self.temporal_context else False:
            return 0.0

        recent = self.temporal_context.confidence_scores[-5:]
        if len(recent) < 2:
            return 0.0

        return current_confidence - recent[0]

    def enhance_summary_for_trajectory(
        self,
        summary: str,
        trajectory: TrajectoryLabel,
        confidence_trend: float,
    ) -> str:
        """Enhance summary text based on trajectory and confidence evolution."""
        if trajectory == "degrading":
            if confidence_trend > 0.2:
                return f"{summary} (confidence increasing; degradation trend accelerating)"
            else:
                return f"{summary} (system degrading)"

        elif trajectory == "improving":
            if confidence_trend > 0.2:
                return f"{summary} (recovering; confirm improvement over next frames)"
            else:
                return f"{summary} (recovery underway)"

        elif trajectory == "unstable":
            return f"{summary} (unstable behavior detected; system oscillating)"

        return summary
