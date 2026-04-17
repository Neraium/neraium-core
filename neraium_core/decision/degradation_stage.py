"""Degradation stage classifier for Phase 4: Degradation Stage Modeling.

Classifies the stage of degradation progression based on temporal context,
persistence, severity, drift velocity, instability, and confidence trends.

Stages:
- baseline: System operating normally, no concerns
- early_shift: Initial changes detected, early warning signals
- emerging_degradation: Clear degradation pattern emerging, not yet persistent
- persistent_degradation: Consistent degradation over multiple frames
- accelerated_deterioration: Degradation is accelerating, high drift velocity
- chronic_degraded_state: Long-term degradation, system has adapted
- failure_approach: Failure conditions imminent, critical thresholds approaching
"""

from __future__ import annotations

from typing import Optional
from neraium_core.decision.models import TemporalContext, SeverityLevel, DegradationStage


def classify_degradation_stage(
    severity: SeverityLevel,
    temporal_context: Optional[TemporalContext],
    persistence_frames: int,
    drift_velocity: float,
    instability: float,
    finding_confidence: float,
) -> DegradationStage:
    """
    Classify the degradation stage based on decision-layer temporal context.

    Args:
        severity: Current severity level (HIGH, ELEVATED, MODERATE, LOW)
        temporal_context: Temporal intelligence context with history
        persistence_frames: Consecutive frames at current severity level
        drift_velocity: Rate of change in drift scores [-1, 1]
        instability: Composite instability measure [0, 1]
        finding_confidence: Confidence in the current finding [0, 1]

    Returns:
        DegradationStage classification
    """
    if temporal_context is None:
        temporal_context = TemporalContext()

    # Extract temporal signals
    severity_history = temporal_context.severity_history[-10:] if temporal_context.severity_history else []
    trajectory = temporal_context.trajectory
    confidence_trend = temporal_context.confidence_trend
    oscillation = temporal_context.oscillation_detected

    # === BASELINE: System is stable with no degradation signals ===
    if severity == "LOW" and persistence_frames < 2:
        if trajectory not in {"degrading", "unstable"}:
            return "baseline"

    # === FAILURE APPROACH: System is in critical condition ===
    # HIGH severity with sustained elevation OR accelerating degradation toward failure
    if severity == "HIGH":
        if persistence_frames >= 2 or (drift_velocity > 0.3 and instability > 0.7):
            return "failure_approach"

    # === ACCELERATED_DETERIORATION: System is degrading rapidly ===
    # ELEVATED/MODERATE with high drift velocity and clear degrading trajectory
    if severity in {"ELEVATED", "MODERATE"}:
        if drift_velocity > 0.25 and trajectory == "degrading" and persistence_frames >= 3:
            return "accelerated_deterioration"

    # === CHRONIC_DEGRADED_STATE: Long-term degradation, system stabilized at lower level ===
    # MODERATE/ELEVATED with sustained persistence but slowing drift
    if severity in {"MODERATE", "ELEVATED"}:
        if persistence_frames >= 8:
            # Check if degradation has slowed (confidence trend negative or oscillating)
            if confidence_trend < -0.1 or oscillation:
                return "chronic_degraded_state"

    # === PERSISTENT_DEGRADATION: Clear sustained degradation ===
    # ELEVATED/MODERATE with 4+ frames at level, low oscillation
    if severity in {"ELEVATED", "MODERATE"}:
        if persistence_frames >= 4 and not oscillation:
            return "persistent_degradation"

    # === EMERGING_DEGRADATION: Degradation pattern appearing ===
    # ELEVATED/MODERATE with 2-3 frames at level, degrading trajectory
    if severity in {"ELEVATED", "MODERATE"}:
        if 2 <= persistence_frames < 4:
            if trajectory in {"degrading", "unstable"} or drift_velocity > 0.15:
                return "emerging_degradation"

    # === EARLY_SHIFT: First signs of departure from baseline ===
    # MODERATE/ELEVATED with low persistence (1-2 frames), indicating newly detected shift
    if severity in {"MODERATE", "ELEVATED"}:
        if persistence_frames <= 2:
            # Presence at MODERATE/ELEVATED itself is a shift signal
            # (elevated from baseline LOW)
            return "early_shift"

    # Default: baseline if no specific stage matches
    return "baseline"


def get_stage_summary(stage: DegradationStage, severity: SeverityLevel) -> str:
    """Get a stage-specific summary message."""
    summaries = {
        "baseline": "System operating at baseline. Structural patterns consistent with normal operation.",
        "early_shift": "Early structural shift detected. Recommend close monitoring for pattern confirmation.",
        "emerging_degradation": "Degradation pattern emerging. System shows signs of structural change.",
        "persistent_degradation": "Persistent degradation confirmed. Sustained change across multiple cycles.",
        "accelerated_deterioration": "Degradation accelerating. System deterioration is increasing in severity.",
        "chronic_degraded_state": "System remains in chronic degraded state. Sustained at reduced performance level.",
        "failure_approach": "Failure approach conditions present. Critical attention required.",
    }
    return summaries.get(stage, "System state unclassified.")


def get_stage_recommendation(stage: DegradationStage) -> dict[str, str]:
    """Get stage-specific recommendation."""
    recommendations = {
        "baseline": {
            "action": "monitor",
            "description": "Continue normal monitoring",
        },
        "early_shift": {
            "action": "monitor_closely",
            "description": "Monitor closely for pattern confirmation",
        },
        "emerging_degradation": {
            "action": "increase_monitoring_cadence",
            "description": "Increase monitoring frequency and review sensor relationships",
        },
        "persistent_degradation": {
            "action": "schedule_inspection",
            "description": "Schedule system inspection and prepare maintenance",
        },
        "accelerated_deterioration": {
            "action": "escalate_to_operations",
            "description": "Escalate to operations team for immediate evaluation",
        },
        "chronic_degraded_state": {
            "action": "sustained_monitoring",
            "description": "Maintain sustained monitoring; assess long-term viability",
        },
        "failure_approach": {
            "action": "immediate_attention",
            "description": "Immediate attention required; prepare for intervention",
        },
    }
    return recommendations.get(stage, {"action": "monitor", "description": "Continue monitoring"})


def should_emit_stage_transition(
    previous_stage: Optional[DegradationStage],
    current_stage: DegradationStage,
) -> bool:
    """Determine if a stage transition should be emitted as an event."""
    if previous_stage is None:
        return False  # Don't emit on first frame

    if previous_stage == current_stage:
        return False  # Same stage, no transition

    # Emit transitions that represent meaningful degradation progression or recovery
    # Skip minor oscillations between adjacent stages
    stage_order = [
        "baseline",
        "early_shift",
        "emerging_degradation",
        "persistent_degradation",
        "accelerated_deterioration",
        "chronic_degraded_state",
        "failure_approach",
    ]

    try:
        prev_idx = stage_order.index(previous_stage)
        curr_idx = stage_order.index(current_stage)
        distance = abs(curr_idx - curr_idx)

        # Emit transitions with distance > 0 (any meaningful change)
        return True
    except ValueError:
        return True  # Unknown stages, emit transition


def format_transition_event(previous_stage: DegradationStage, current_stage: DegradationStage) -> str:
    """Format a stage transition event string."""
    return f"stage_transition:{previous_stage}→{current_stage}"
