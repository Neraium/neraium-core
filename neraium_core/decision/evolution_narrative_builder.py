"""Evolution narrative builder: Generate human-readable evolution explanations.

Constructs the story of how a system has evolved to its current state,
explaining both the path and the mechanisms driving change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from neraium_core.decision.models import EvolutionContext, SeverityLevel
from neraium_core.decision.causal_degradation_tracker import DegradationMap


@dataclass
class EvolutionNarrative:
    """Human-readable evolution story."""

    entry_story: str  # How we got to current regime
    current_state_story: str  # What's happening now
    trajectory_story: str  # Direction & velocity
    evidence_summary: str  # Most important signals
    projection_story: str  # Where headed if unchanged
    inflection_points: list[str] = None  # Critical moments in evolution

    def __post_init__(self):
        if self.inflection_points is None:
            self.inflection_points = []


class EvolutionNarrativeBuilder:
    """Builds human-readable narratives from evolution context."""

    def __init__(self):
        pass

    def build_narrative(
        self,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        severity: SeverityLevel,
        drift_score: float,
        finding_confidence: float,
        persistence_frames: int,
    ) -> EvolutionNarrative:
        """Build complete evolution narrative from context.

        Args:
            evolution_context: Evolution tracking context
            degradation_map: Relationship degradation analysis
            severity: Current severity level
            drift_score: Structural drift [0, 1]
            finding_confidence: Confidence in findings [0, 1]
            persistence_frames: Frames at current severity

        Returns:
            EvolutionNarrative with story components
        """
        if not evolution_context:
            return EvolutionNarrative(
                entry_story="No evolution history.",
                current_state_story="System state unknown.",
                trajectory_story="No trajectory data.",
                evidence_summary="Insufficient data.",
                projection_story="Cannot project.",
            )

        # Build story components
        entry_story = self._build_entry_story(evolution_context, persistence_frames)
        current_state_story = self._build_current_state_story(
            evolution_context, severity, degradation_map
        )
        trajectory_story = self._build_trajectory_story(evolution_context, drift_score)
        evidence_summary = self._build_evidence_summary(
            evolution_context, finding_confidence, degradation_map
        )
        projection_story = self._build_projection_story(
            evolution_context, severity, degradation_map
        )
        inflection_points = self._identify_inflection_points(evolution_context)

        return EvolutionNarrative(
            entry_story=entry_story,
            current_state_story=current_state_story,
            trajectory_story=trajectory_story,
            evidence_summary=evidence_summary,
            projection_story=projection_story,
            inflection_points=inflection_points,
        )

    def _build_entry_story(
        self, evolution_context: EvolutionContext, persistence_frames: int
    ) -> str:
        """How did we enter the current regime?"""
        if not evolution_context.regime_sequence or len(evolution_context.regime_sequence) < 2:
            return f"In {evolution_context.current_regime} regime (initial state)."

        if len(evolution_context.regime_sequence) == 2:
            prev_regime = evolution_context.regime_sequence[-2]
            return f"Transitioned from {prev_regime} to {evolution_context.current_regime} {persistence_frames} frames ago."

        # Multi-step transition
        regime_path = " → ".join(evolution_context.regime_sequence[-3:])
        return f"Progression: {regime_path}. Currently {persistence_frames} frames in {evolution_context.current_regime}."

    def _build_current_state_story(
        self,
        evolution_context: EvolutionContext,
        severity: SeverityLevel,
        degradation_map: Optional[DegradationMap],
    ) -> str:
        """What's happening right now?"""
        stories = []

        # Regime status
        regime_desc = {
            "STABLE": "operating normally",
            "WATCH": "showing elevated activity",
            "ALERT": "exhibiting degradation signals",
            "CRITICAL": "in critical condition",
        }
        stories.append(
            f"Currently {regime_desc.get(evolution_context.current_regime, 'in transition')}"
        )

        # Movement and coherence
        if evolution_context.is_coherent:
            signal_quality = "persistent, coherent signal"
        else:
            signal_quality = "inconsistent, noisy signal"

        stories.append(f"with {signal_quality}")

        # Degradation picture
        if degradation_map and degradation_map.degradation_count > 0:
            stories.append(
                f"({degradation_map.degradation_count} relationship(s) degrading)"
            )

        return "; ".join(stories) + "."

    def _build_trajectory_story(
        self, evolution_context: EvolutionContext, drift_score: float
    ) -> str:
        """Where is the system heading?"""
        direction_map = {
            "degrading": "deteriorating",
            "improving": "recovering",
            "stable": "holding",
        }
        direction = direction_map.get(evolution_context.trajectory_direction, "unclear")

        pattern_map = {
            "linear": "steadily",
            "oscillating": "erratically",
            "accelerating": "rapidly",
        }
        pattern = pattern_map.get(evolution_context.movement_pattern, "")

        speed = "slowly" if drift_score < 0.3 else "moderately" if drift_score < 0.6 else "quickly"

        if pattern:
            return f"System {direction} {pattern} (drift: {speed})."
        else:
            return f"System {direction} {speed}."

    def _build_evidence_summary(
        self,
        evolution_context: EvolutionContext,
        finding_confidence: float,
        degradation_map: Optional[DegradationMap],
    ) -> str:
        """Most important signals backing the assessment."""
        signals = []

        # Temporal coherence
        if evolution_context.temporal_coherence > 0.7:
            signals.append("strong temporal coherence")
        elif evolution_context.temporal_coherence > 0.5:
            signals.append("moderate consistency")

        # Persistence
        if evolution_context.regime_persistence >= 5:
            signals.append(f"sustained ({evolution_context.regime_persistence} frames)")

        # Degradation count
        if degradation_map and degradation_map.critical_degradation_count > 0:
            signals.append(f"{degradation_map.critical_degradation_count} critical relationship(s) failing")

        # Confidence
        if finding_confidence > 0.75:
            signals.append("high confidence")

        # Transitions
        if evolution_context.regime_transitions_count > 2:
            signals.append(f"multiple transitions ({evolution_context.regime_transitions_count})")

        if not signals:
            return "Limited evidence."

        return " | ".join(signals) + "."

    def _build_projection_story(
        self,
        evolution_context: EvolutionContext,
        severity: SeverityLevel,
        degradation_map: Optional[DegradationMap],
    ) -> str:
        """If trends continue, what happens next?"""
        if not evolution_context.is_degrading:
            return "No deterioration trend projected."

        severity_rank = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}
        current_rank = severity_rank.get(severity, 1)

        if evolution_context.movement_pattern == "accelerating":
            if current_rank < 3:
                return f"Without intervention, expect escalation to next severity level within 3-5 frames."
            else:
                return "Critical phase: system at highest severity with accelerating degradation."

        elif evolution_context.movement_pattern == "oscillating":
            return "Oscillating instability detected; unpredictable transitions possible."

        else:  # linear
            frames_to_next = max(3, 10 - evolution_context.regime_persistence)
            if current_rank < 3:
                return f"Linear degradation suggests {frames_to_next} frames to next severity level."
            else:
                return "Sustained critical state; continued monitoring essential."

    def _identify_inflection_points(self, evolution_context: EvolutionContext) -> list[str]:
        """Identify critical moments in evolution."""
        inflections = []

        # Multiple rapid transitions
        transition_points = getattr(evolution_context, "transition_points", [])
        regime_sequence = getattr(evolution_context, "regime_sequence", [])
        if len(transition_points) >= 2 and len(regime_sequence) >= 3:
            recent_transitions = transition_points[-2:]
            if len(recent_transitions) == 2:
                delta = recent_transitions[1] - recent_transitions[0]
                if delta <= 5:
                    inflections.append(f"Rapid regime transitions (every {delta} frames)")

        # Accelerating movement
        if evolution_context.movement_pattern == "accelerating":
            inflections.append("Acceleration detected: degradation rate increasing")

        # Oscillation onset
        if evolution_context.movement_pattern == "oscillating":
            inflections.append("Oscillation pattern: instability regime")

        # Long persistence
        if evolution_context.regime_persistence >= 10:
            inflections.append(
                f"Extended persistence in {evolution_context.current_regime} regime ({evolution_context.regime_persistence} frames)"
            )

        return inflections

    def format_for_decision_trace(self, narrative: EvolutionNarrative) -> str:
        """Format narrative concisely for decision_trace integration."""
        parts = []

        if narrative.entry_story:
            parts.append(narrative.entry_story)

        if narrative.trajectory_story:
            parts.append(narrative.trajectory_story)

        if narrative.inflection_points:
            inflection_summary = "; ".join(narrative.inflection_points[:2])
            parts.append(f"Key events: {inflection_summary}")

        return " ".join(parts)
