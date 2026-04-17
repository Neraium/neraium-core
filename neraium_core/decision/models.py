"""Data models for the decision layer.

Structures that flow through the decision pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


SeverityLevel = Literal["CRITICAL", "HIGH", "MODERATE", "LOW"]


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
        }
