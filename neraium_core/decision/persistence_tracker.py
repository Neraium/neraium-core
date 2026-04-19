"""Track persistence of severity levels across frames for hysteresis.

Implements escalation/suppression logic based on sustained evidence.
"""

from __future__ import annotations

from typing import Optional
from neraium_core.decision.models import SeverityLevel, PersistenceState


class PersistenceTracker:
    """Tracks severity persistence across frames for hysteresis."""

    # Minimum consecutive frames at each severity before escalating up
    ESCALATION_REQUIREMENTS = {
        "LOW": 0,
        "MODERATE": 1,
        "ELEVATED": 2,
        "HIGH": 0,  # Once at HIGH, require stronger evidence to stay
    }

    # Minimum frames to suppress at startup/transient
    STARTUP_SUPPRESS_WINDOW = 8

    def __init__(self):
        pass

    def update_frame(
        self,
        current_severity: SeverityLevel,
        previous_severity: SeverityLevel | None,
        confidence: float,
        state: str,
        frame_number: int,
    ) -> PersistenceState:
        """Update tracking for new frame.

        Args:
            current_severity: The newly classified severity
            previous_severity: Severity from prior frame
            confidence: Finding confidence for this frame
            state: System state (STABLE, WATCH, ALERT, etc)
            frame_number: Current frame number

        Returns:
            Updated PersistenceState
        """
        if previous_severity is None:
            previous_severity = "LOW"

        # Create or update persistence state
        state_obj = PersistenceState()
        state_obj.current_level = current_severity

        # Update consecutive frame counter
        if current_severity == previous_severity:
            state_obj.consecutive_frames_at_level[current_severity] = (
                state_obj.consecutive_frames_at_level.get(current_severity, 0) + 1
            )
        else:
            state_obj.consecutive_frames_at_level = {
                "LOW": 0,
                "ELEVATED": 0,
                "MODERATE": 0,
                "HIGH": 0,
            }
            state_obj.consecutive_frames_at_level[current_severity] = 1

        # Track history (last 10 levels)
        state_obj.level_history.append(current_severity)
        if len(state_obj.level_history) > 10:
            state_obj.level_history = state_obj.level_history[-10:]

        return state_obj

    def can_escalate_to(
        self,
        from_level: SeverityLevel,
        to_level: SeverityLevel,
        current_frames: int,
    ) -> bool:
        """Check if we have sufficient frames at current level to escalate.

        Implements hysteresis: requires sustained evidence before escalating.
        """
        # HIGH → any: always possible (HIGH is strongest)
        if from_level == "HIGH":
            return False

        # Escalation always requires frames at current level
        required = self.ESCALATION_REQUIREMENTS.get(from_level, 1)
        return current_frames >= required

    def should_suppress_startup(
        self,
        severity: SeverityLevel,
        frame_number: int,
        transient_score: float,
        consecutive_frames: int,
    ) -> bool:
        """Apply startup/transient suppression.

        Never suppresses HIGH or after warmup.
        """
        # HIGH is never suppressed
        if severity == "HIGH":
            return False

        # After startup window, allow through if persistent
        if frame_number >= self.STARTUP_SUPPRESS_WINDOW and consecutive_frames >= 2:
            return False

        # During startup window
        if frame_number < self.STARTUP_SUPPRESS_WINDOW:
            # ELEVATED/MODERATE: suppress until confirmed (2+ frames)
            if severity in {"ELEVATED", "MODERATE"} and consecutive_frames < 2:
                return True
            # Transient MODERATE with high transience score: suppress
            if severity == "MODERATE" and transient_score > 0.65:
                return True

        return False

    def compute_trajectory(self, level_history: list[SeverityLevel]) -> float:
        """Compute severity trajectory: -1 improving, 0 stable, +1 degrading.

        Based on recent history.
        """
        if len(level_history) < 2:
            return 0.0

        # Map levels to numeric scores
        level_score = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}

        recent = [level_score.get(l, 0) for l in level_history[-5:]]
        if len(recent) < 2:
            return 0.0

        # Simple trend: last vs first
        trend = (recent[-1] - recent[0]) / (len(recent) - 1)

        # Clamp to [-1, 1]
        return max(-1.0, min(1.0, trend))

    def should_emit_new_recommendation(
        self,
        new_action: str,
        last_action: str,
        frames_since_last: int,
        severity_escalated: bool,
    ) -> bool:
        """Determine if we should emit a new recommendation.

        Avoids repeating same recommendation every frame.
        """
        # Different action: always emit
        if new_action != last_action:
            return True

        # Same action: only re-emit if escalated or enough frames passed
        if severity_escalated:
            return True

        # Re-emit every 5 frames to keep operator informed
        if frames_since_last >= 5:
            return True

        return False

    def get_suppression_reason(
        self,
        suppress: bool,
        is_first_appearance: bool,
        is_safe_transient: bool,
        frame_number: int,
    ) -> str:
        """Generate human-readable suppression reason."""
        if not suppress:
            return ""

        if frame_number < self.STARTUP_SUPPRESS_WINDOW:
            return "Startup phase (waiting for stability confirmation)"

        if is_first_appearance:
            return "First appearance (requires sustained evidence)"

        if is_safe_transient:
            return "Matches known safe transient pattern"

        return "Low confidence/transient (below emission threshold)"
