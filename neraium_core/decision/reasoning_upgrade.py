"""Decision reasoning upgrade: improved causal abstraction, temporal judgment, uncertainty handling.

Implements three core improvements:
1. Causal abstraction - collapses signal-level artifacts into subsystem/relational themes
2. Temporal judgment - classifies phase: onset, persistent, accelerating, chronic, resolving
3. Uncertainty handling - distinguishes confidence levels under partial evidence
"""

from __future__ import annotations

from typing import Literal

CausalTheme = Literal[
    "relational_breakdown",
    "persistent_structural_degradation",
    "multi_signal_instability",
    "transient_anomaly",
    "external_shock",
]

TemporalPhase = Literal[
    "onset",
    "persistent",
    "accelerating",
    "chronic",
    "resolving",
]

UncertaintyContext = Literal[
    "low_evidence",
    "conflicting_signals",
    "unstable_trajectory",
    "high_confidence",
]


def derive_causal_theme(
    drift: float,
    relational_stability: float,
    coherence: float,
    persistence_frames: int,
    trajectory: str,
) -> CausalTheme:
    """
    Derive causal abstraction from subsystem-level metrics.

    Inputs:
        drift: [0, 1] structural drift score
        relational_stability: [0, 1] quality of relational coherence
        coherence: [0, 1] system coherence metric
        persistence_frames: count of consecutive frames at elevated level
        trajectory: "stable" | "degrading" | "recovering" | "unstable"

    Returns:
        Primary causal theme explaining the root cause of degradation
    """
    # External shock: sudden high drift with unstable trajectory, low persistence
    if drift > 0.7 and trajectory == "unstable" and persistence_frames <= 2:
        return "external_shock"

    # Relational breakdown: coherence loss despite moderate drift
    if relational_stability < 0.4 and coherence < 0.50:
        return "relational_breakdown"

    # Multi-signal instability: elevated drift with degrading trajectory and some persistence
    if drift > 0.5 and trajectory == "degrading" and persistence_frames >= 3:
        return "multi_signal_instability"

    # Persistent structural degradation: sustained high drift with poor coherence
    if drift > 0.6 and coherence < 0.55 and persistence_frames >= 5:
        return "persistent_structural_degradation"

    # Transient anomaly: low relational stability but recovering or stable trajectory
    if relational_stability < 0.50 and trajectory in {"stable", "recovering"}:
        return "transient_anomaly"

    # Default: structural degradation if any significant drift
    if drift > 0.5:
        return "multi_signal_instability"

    return "transient_anomaly"


def classify_temporal_phase(
    persistence_frames: int,
    drift_velocity: float,
    trajectory: str,
    coherence_trend: float,
) -> TemporalPhase:
    """
    Classify the temporal progression phase of degradation.

    Inputs:
        persistence_frames: consecutive frames at current severity level
        drift_velocity: rate of drift change [-1, 1]
        trajectory: "stable" | "degrading" | "recovering" | "unstable"
        coherence_trend: change in coherence over recent frames [-1, 1]

    Returns:
        Classification of progression phase: onset, persistent, accelerating, chronic, resolving
    """
    # Onset: first 1-2 frames of deviation
    if persistence_frames <= 2 and trajectory == "degrading":
        return "onset"

    # Resolving: improving trajectory, coherence recovering
    if trajectory == "recovering" or coherence_trend > 0.2:
        return "resolving"

    # Accelerating: increasing drift velocity or worsening trajectory
    if drift_velocity > 0.1 and trajectory == "degrading":
        return "accelerating"

    # Chronic: long persistence with reduced acceleration
    if persistence_frames >= 10 and drift_velocity <= 0.05:
        return "chronic"

    # Persistent: 3-6 frames sustained at same level
    if 3 <= persistence_frames <= 6 and trajectory == "degrading":
        return "persistent"

    # Default to persistent for moderate persistence
    if persistence_frames >= 3:
        return "persistent"

    return "onset"


def classify_uncertainty_context(
    finding_confidence: float,
    persistence_frames: int,
    trajectory: str,
    pattern_match_tier: str | None,
    causal_chain_strength: float,
) -> UncertaintyContext:
    """
    Classify the confidence and evidence context for decision-making.

    Inputs:
        finding_confidence: [0, 1] confidence something actually happened
        persistence_frames: consecutive elevated frames
        trajectory: "stable" | "degrading" | "recovering" | "unstable"
        pattern_match_tier: "weak" | "moderate" | "strong" | None
        causal_chain_strength: [0, 1] strength of causal reasoning

    Returns:
        Uncertainty context classification
    """
    # High confidence: strong persistence, degrading trajectory, good pattern match/causal
    if (
        finding_confidence > 0.75
        and persistence_frames >= 3
        and trajectory == "degrading"
        and (pattern_match_tier == "strong" or causal_chain_strength > 0.7)
    ):
        return "high_confidence"

    # Unstable trajectory: contradictory signals or oscillating state
    if trajectory == "unstable":
        return "unstable_trajectory"

    # Conflicting signals: moderate confidence but uncertain trajectory
    if finding_confidence > 0.5 and persistence_frames <= 2 and trajectory != "degrading":
        return "conflicting_signals"

    # Low evidence: insufficient data or weak signals
    if finding_confidence < 0.5 or (persistence_frames <= 1 and pattern_match_tier != "strong"):
        return "low_evidence"

    return "high_confidence" if finding_confidence > 0.6 else "conflicting_signals"


def format_causal_summary(causal_theme: CausalTheme) -> str:
    """Convert causal theme to human-readable summary."""
    theme_descriptions = {
        "relational_breakdown": "Relational breakdown is the primary driver",
        "persistent_structural_degradation": "Persistent structural degradation detected",
        "multi_signal_instability": "Multi-signal instability observed",
        "transient_anomaly": "Transient anomaly detected (may self-resolve)",
        "external_shock": "External shock or sudden change detected",
    }
    return theme_descriptions.get(causal_theme, "System degradation detected")


def format_temporal_context(
    temporal_phase: TemporalPhase,
    persistence_frames: int,
) -> str:
    """Convert temporal phase and persistence to human-readable context."""
    if temporal_phase == "onset":
        return "Signal emerging but not yet persistent"
    elif temporal_phase == "persistent":
        return f"Persistent degradation confirmed ({persistence_frames} frames)"
    elif temporal_phase == "accelerating":
        return "Acceleration detected (drift increasing)"
    elif temporal_phase == "chronic":
        return "Chronic degradation with stable drift"
    elif temporal_phase == "resolving":
        return "Trajectory improving; monitoring for confirmation"
    return "Temporal pattern monitoring"


def format_uncertainty_rationale(
    uncertainty_context: UncertaintyContext,
    finding_confidence: float,
    persistence_frames: int,
) -> str:
    """Generate brief rationale explaining confidence and evidence quality."""
    if uncertainty_context == "high_confidence":
        if persistence_frames >= 5:
            return "High confidence (multi-frame sustained signal)"
        else:
            return "High confidence (strong signal quality)"
    elif uncertainty_context == "low_evidence":
        return "Pattern match weak; relying on live evidence"
    elif uncertainty_context == "conflicting_signals":
        return "Signal emerging but trajectory unstable; monitoring"
    elif uncertainty_context == "unstable_trajectory":
        return "Trajectory unstable; multiple interpretations possible"
    return "Moderate confidence, continuing observation"
