"""Data models for the decision layer.

Structures that flow through the decision pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


SeverityLevel = Literal["HIGH", "ELEVATED", "MODERATE", "LOW"]
TrajectoryLabel = Literal["improving", "stable", "degrading", "unstable"]
DegradationStage = Literal[
    "baseline",
    "early_shift",
    "emerging_degradation",
    "persistent_degradation",
    "accelerated_deterioration",
    "chronic_degraded_state",
    "failure_approach",
]


@dataclass
class TemporalContext:
    """Track temporal patterns and drift behavior across recent frames."""
    severity_history: list[str] = field(default_factory=list)  # Last 5-10 severity levels
    drift_scores: list[float] = field(default_factory=list)  # Last 5-10 drift scores
    confidence_scores: list[float] = field(default_factory=list)  # Confidence evolution
    trajectory: TrajectoryLabel = "stable"  # Current trajectory direction
    drift_velocity: float = 0.0  # Rate of drift change [-1, 1]
    oscillation_detected: bool = False  # Whether severity is oscillating
    persistent_frames_at_level: int = 0  # Consecutive frames at current level
    last_recommendation: Optional[str] = None  # Last emitted recommendation
    last_recommendation_frame: int = -1000  # When last recommendation was made
    confidence_trend: float = 0.0  # Is confidence increasing/decreasing [-1, 1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PersistenceState:
    """Track consecutive frames at each severity level for hysteresis logic."""
    consecutive_frames_at_level: dict[str, int] = field(
        default_factory=lambda: {"HIGH": 0, "ELEVATED": 0, "MODERATE": 0, "LOW": 0}
    )
    level_history: list[str] = field(default_factory=list)
    current_level: str = "LOW"
    suppression_streak: int = 0
    last_recommendation_frame: int = -1000
    last_recommendation_action: str = ""
    last_recommendation_id: str = ""
    suppression_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Finding:
    """A specific observable change in the system."""
    category: str  # e.g., "correlation_loss", "subsystem_instability", "signal_degradation"
    description: str  # e.g., "pressure-temp correlation dropped 0.23"
    confidence: float  # [0, 1] how confident this change is real
    magnitude: float  # [0, 1] how large is the change
    reversible: Optional[bool] = None  # Can the system recover from this?
    affected_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalStep:
    """One link in a causal chain."""
    trigger: str  # What happened (e.g., "vibration_spike")
    effect: str  # What resulted (e.g., "pressure_coordination_loss")
    strength: float  # [0, 1] how strong is this linkage
    involved_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CausalChain:
    """Chain of causation explaining how we got to current state."""
    steps: list[CausalStep] = field(default_factory=list)
    root_cause: Optional[str] = None
    confidence: float = 0.5  # How confident in this chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "root_cause": self.root_cause,
            "confidence": self.confidence,
        }


@dataclass
class PatternMatch:
    """Match to a known prior pattern."""
    pattern_id: str  # e.g., "asset_pump_A0_001:run_42"
    similarity: float  # [0, 1] cosine similarity
    prior_outcome: str  # e.g., "self_resolved", "escalated_to_failure"
    time_to_outcome_hours: Optional[float] = None
    confidence: float = 0.7
    # Phase 5 additions
    match_tier: Optional[str] = None  # "weak", "moderate", "strong"
    outcome_type: Optional[str] = None  # "self_resolved" | "persistent_degradation" | "failure_progression" | "unknown"
    confidence_weight: float = 1.0  # [0, 1] how much to weight this pattern in decisions
    stage_at_match: Optional[str] = None  # degradation stage when pattern was originally observed

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Recommendation:
    """Advisory recommendation for human operator."""
    action: str  # e.g., "monitor", "inspect", "schedule_maintenance", "increase_cadence"
    target: Optional[str] = None  # What to focus on (signal/subsystem)
    urgency: SeverityLevel = "MODERATE"
    rationale: str = ""
    time_window_hours: Optional[float] = None
    confidence: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """The decision layer output: what to surface, what confidence, what to recommend."""

    # Core confidence metrics
    finding_confidence: float  # [0, 1] confidence something actually happened
    action_confidence: float  # [0, 1] confidence in the recommendation
    transient_score: float  # [0, 1] likelihood this is temporary

    # What to do with this finding
    suppress: bool  # Should we hide this from operators?
    severity: SeverityLevel

    # Human-readable summary
    summary: str

    # Detailed findings
    findings: list[Finding] = field(default_factory=list)

    # Causal explanation
    causal_chain: Optional[CausalChain] = None

    # Historical pattern matching
    pattern_match: Optional[PatternMatch] = None

    # What to do about it
    recommended_action: Optional[str] = None
    recommended_target: Optional[str] = None

    # Why we made this decision
    reasons: list[str] = field(default_factory=list)

    # Persistence tracking state (for hysteresis across frames)
    persistence_state: Optional[PersistenceState] = None

    # Persistence context
    persistence_frames_at_level: int = 0  # How many consecutive frames at current severity
    is_first_appearance: bool = False  # First frame at this severity level

    # Phase 3: Temporal Intelligence
    trajectory: TrajectoryLabel = "stable"  # Direction of severity evolution
    temporal_confidence_delta: float = 0.0  # Change in confidence over last few frames [-1, 1]
    transition_event: Optional[str] = None  # Only set when real state transition occurs
    temporal_context: Optional[TemporalContext] = None  # Full temporal context for this frame
    consistency_check_passed: bool = True  # Whether decision is consistent with recent history

    # Phase 4: Degradation Stage Modeling
    degradation_stage: DegradationStage = "baseline"  # Stage of degradation progression
    stage_transition_event: Optional[str] = None  # Stage transition marker (e.g., "stage_transition:baseline→early_shift")
    stage_specific_recommendation: Optional[str] = None  # Stage-specific action recommendation

    # Phase 5: Outcome-Aware Decision Intelligence
    pattern_outcome_type: Optional[str] = None  # "self_resolved" | "persistent_degradation" | "failure_progression" | "unknown"
    pattern_match_tier: Optional[str] = None  # "weak", "moderate", "strong"
    pattern_influence_summary: Optional[str] = None  # Explanation of how patterns influenced decision

    # Phase 6: Multi-Horizon Action Intelligence
    action_horizon: Optional[str] = None  # "now" | "soon" | "watchlist"
    primary_action: Optional[str] = None  # Primary recommended action
    secondary_actions: list[str] = field(default_factory=list)  # Supporting actions
    action_priority_reason: Optional[str] = None  # Why this horizon was chosen
    action_tradeoff_note: Optional[str] = None  # Tradeoffs in action selection (e.g., deferred due to cost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_confidence": self.finding_confidence,
            "action_confidence": self.action_confidence,
            "transient_score": self.transient_score,
            "suppress": self.suppress,
            "severity": self.severity,
            "summary": self.summary,
            "findings": [f.to_dict() for f in self.findings],
            "causal_chain": self.causal_chain.to_dict() if self.causal_chain else None,
            "pattern_match": self.pattern_match.to_dict() if self.pattern_match else None,
            "recommended_action": self.recommended_action,
            "recommended_target": self.recommended_target,
            "reasons": self.reasons,
            "persistence_state": self.persistence_state.to_dict() if self.persistence_state else None,
            "persistence_frames_at_level": self.persistence_frames_at_level,
            "is_first_appearance": self.is_first_appearance,
            "trajectory": self.trajectory,
            "temporal_confidence_delta": self.temporal_confidence_delta,
            "transition_event": self.transition_event,
            "temporal_context": self.temporal_context.to_dict() if self.temporal_context else None,
            "consistency_check_passed": self.consistency_check_passed,
            "degradation_stage": self.degradation_stage,
            "stage_transition_event": self.stage_transition_event,
            "stage_specific_recommendation": self.stage_specific_recommendation,
            "pattern_outcome_type": self.pattern_outcome_type,
            "pattern_match_tier": self.pattern_match_tier,
            "pattern_influence_summary": self.pattern_influence_summary,
            "action_horizon": self.action_horizon,
            "primary_action": self.primary_action,
            "secondary_actions": self.secondary_actions,
            "action_priority_reason": self.action_priority_reason,
            "action_tradeoff_note": self.action_tradeoff_note,
        }
