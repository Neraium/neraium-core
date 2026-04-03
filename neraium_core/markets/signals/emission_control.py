"""Noise suppression and hysteresis controls for operator-facing outputs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EmissionDecision:
    emit: bool
    cooldown_status: str | None = None


@dataclass(slots=True)
class EmissionControlConfig:
    min_confidence: float = 0.45
    cooldown_frames: int = 2
    duplicate_tolerance: float = 0.04
    hysteresis_margin: float = 0.06
    emit_state_changes_only: bool = False
    warmup_bars: int = 80


@dataclass(slots=True)
class _EmissionMemory:
    last_permission: str | None = None
    last_best_action: str | None = None
    last_validity_score: float | None = None
    last_market_state: str | None = None
    last_transition_state: str | None = None
    last_emitted_index: int = -10_000


@dataclass(slots=True)
class SignalEmissionController:
    config: EmissionControlConfig = field(default_factory=EmissionControlConfig)
    _memory: dict[str, _EmissionMemory] = field(default_factory=dict)

    def stabilize_permission(self, key: str, action_permission: str, validity_score: float) -> str:
        memory = self._memory.setdefault(key, _EmissionMemory())
        if memory.last_permission is None or memory.last_validity_score is None:
            return action_permission
        if action_permission != memory.last_permission and abs(validity_score - memory.last_validity_score) < self.config.hysteresis_margin:
            return memory.last_permission
        return action_permission

    def evaluate(
        self,
        key: str,
        *,
        frame_index: int,
        sample_count: int,
        confidence: float,
        action_permission: str,
        best_action: str,
        validity_score: float,
        market_state: str,
        transition_state: str,
        abstain: bool,
    ) -> EmissionDecision:
        memory = self._memory.setdefault(key, _EmissionMemory())

        if sample_count < self.config.warmup_bars:
            return EmissionDecision(False, "warmup")
        if confidence < self.config.min_confidence and abstain:
            if memory.last_emitted_index < 0:
                return EmissionDecision(True, "low_confidence_abstention")
            return EmissionDecision(False, "low_confidence_abstention")

        changed = any(
            [
                memory.last_permission != action_permission,
                memory.last_best_action != best_action,
                memory.last_market_state != market_state,
                memory.last_transition_state != transition_state,
            ]
        )
        duplicate = (
            memory.last_permission == action_permission
            and memory.last_best_action == best_action
            and memory.last_validity_score is not None
            and abs(memory.last_validity_score - validity_score) < self.config.duplicate_tolerance
        )
        in_cooldown = frame_index - memory.last_emitted_index <= self.config.cooldown_frames

        if self.config.emit_state_changes_only and not changed:
            return EmissionDecision(False, "state_unchanged")
        if duplicate and in_cooldown:
            return EmissionDecision(False, "duplicate")
        if in_cooldown and not changed:
            return EmissionDecision(False, "cooldown")
        return EmissionDecision(True, "state_change" if changed else "ready")

    def mark_emitted(
        self,
        key: str,
        *,
        frame_index: int,
        action_permission: str,
        best_action: str,
        validity_score: float,
        market_state: str,
        transition_state: str,
    ) -> None:
        memory = self._memory.setdefault(key, _EmissionMemory())
        memory.last_permission = action_permission
        memory.last_best_action = best_action
        memory.last_validity_score = validity_score
        memory.last_market_state = market_state
        memory.last_transition_state = transition_state
        memory.last_emitted_index = frame_index
