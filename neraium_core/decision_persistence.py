"""Decision persistence and hysteresis mechanism.

Ensures the decision engine commits to a path rather than flickering.
Implements 20% threshold for action switching and momentum-based confidence.
"""

from __future__ import annotations

from typing import Any


class DecisionCommitment:
    """Tracks primary action commitment with hysteresis and momentum."""

    def __init__(self, hysteresis_threshold: float = 0.20, max_momentum: float = 1.0):
        """
        Args:
            hysteresis_threshold: Score delta needed to switch actions (default 20%)
            max_momentum: Maximum confidence boost from sustained decisions (default 1.0)
        """
        self.hysteresis_threshold = hysteresis_threshold
        self.max_momentum = max_momentum

        # State
        self.active_action: dict[str, Any] | None = None
        self.active_score: float = 0.0
        self.consecutive_cycles: int = 0
        self.momentum: float = 0.0  # [0, max_momentum]

    def evaluate(
        self, candidate_action: dict[str, Any], candidate_score: float
    ) -> tuple[dict[str, Any] | None, float]:
        """
        Evaluate whether to switch actions or stay committed.

        Args:
            candidate_action: Proposed new action
            candidate_score: Score of proposed action [0, 1]

        Returns:
            (committed_action, confidence_boost)
            - committed_action: The action to execute (or None if no change)
            - confidence_boost: Momentum-based confidence adjustment
        """
        # First evaluation: establish commitment
        if self.active_action is None:
            self.active_action = candidate_action.copy()
            self.active_score = candidate_score
            self.consecutive_cycles = 1
            self.momentum = 0.0
            return self.active_action, 0.0

        # Same action: increase momentum
        if self._is_same_action(self.active_action, candidate_action):
            self.consecutive_cycles += 1
            # Momentum grows but caps at max_momentum
            self.momentum = min(self.max_momentum, self.momentum + 0.1)
            return None, self.momentum  # No action switch, but return momentum

        # Different action: check hysteresis threshold
        score_delta = candidate_score - self.active_score
        requires_switch = score_delta > self.hysteresis_threshold

        if requires_switch:
            # Strong enough to switch: reset commitment state
            self.active_action = candidate_action.copy()
            self.active_score = candidate_score
            self.consecutive_cycles = 1
            self.momentum = 0.0  # Reset momentum on switch
            return self.active_action, 0.0

        # Not strong enough: stay committed to current action
        self.consecutive_cycles += 1
        return None, self.momentum

    def get_active_action(self) -> dict[str, Any] | None:
        """Get the currently committed action."""
        return self.active_action.copy() if self.active_action else None

    def get_momentum(self) -> float:
        """Get current momentum [0, max_momentum]."""
        return self.momentum

    def reset(self):
        """Clear commitment state."""
        self.active_action = None
        self.active_score = 0.0
        self.consecutive_cycles = 0
        self.momentum = 0.0

    @staticmethod
    def _is_same_action(action1: dict[str, Any], action2: dict[str, Any]) -> bool:
        """Check if two actions are logically equivalent."""
        return action1.get("action_type") == action2.get("action_type")


def apply_momentum_to_confidence(base_confidence: float, momentum: float) -> float:
    """Apply momentum boost to base confidence score.

    High momentum means the action has been consistently dominant.
    Boost is applied gradually to avoid perceptual jump.

    Args:
        base_confidence: Base confidence [0, 1]
        momentum: Momentum from decision commitment [0, 1]

    Returns:
        Enhanced confidence score
    """
    # Momentum compounds confidence, but doesn't override low base scores
    boost = momentum * 0.25  # Max 25% confidence boost from momentum
    return min(1.0, base_confidence + boost)
