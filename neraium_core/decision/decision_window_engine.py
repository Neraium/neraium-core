"""Decision Window Engine: Quantify WHEN intervention becomes ineffective.

Core insight: Early detection is only valuable if operators know WHEN they must act.
This engine computes:
- Time remaining before intervention stops working
- Effectiveness curve (intervention impact over the window)
- Irreversibility detection (when recovery becomes impossible)
- Window confidence (backed by temporal coherence)

Output: "ACT NOW: 4-6 hours remaining | confidence 0.82"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
from neraium_core.decision.models import SeverityLevel, DegradationStage, EvolutionContext
from neraium_core.decision.causal_degradation_tracker import DegradationMap


WindowStatus = Literal["OPEN", "NARROWING", "CRITICAL", "CLOSED"]
InterventionPhase = Literal["EARLY", "MID", "LATE", "FINAL"]


@dataclass
class InterventionWindow:
    """Time window during which intervention is still effective."""

    status: WindowStatus  # OPEN | NARROWING | CRITICAL | CLOSED
    estimated_hours_remaining: Optional[float]  # Best estimate of hours left
    confidence: float  # 0-1, backed by temporal coherence
    phase: InterventionPhase  # EARLY | MID | LATE | FINAL
    effectiveness: float  # Current intervention impact [0, 1]
    irreversible_at_frame: Optional[int]  # Frame when recovery becomes impossible
    narrowing_rate: str  # "slow" | "moderate" | "rapid" | "critical"
    recommended_action_urgency: str  # "monitor" | "prepare" | "act_now" | "impossible"


@dataclass
class DecisionWindow:
    """Full window analysis for a frame."""

    intervention_window: InterventionWindow
    why_closing: list[str]  # Evidence for closure (e.g., "accelerating degradation", "critical relationships failing")
    if_wait_one_hour: str  # Projection: "window shrinks from 6h to 3h"
    if_wait_next_stage: str  # Projection: "next stage closes window by 40%"


class DecisionWindowEngine:
    """Computes intervention time windows based on evolution signals."""

    # Window definitions (in frame units, assume ~1 frame = 5-10 minutes)
    FRAME_TO_HOURS = 0.083  # ~1 frame = 5 minutes = 0.083 hours

    # Stage-based baseline windows (how many frames until lock-in from each stage)
    STAGE_WINDOWS = {
        "baseline": 1000,  # Can wait; structure stable
        "early_shift": 500,  # Early warning; 40+ hours
        "emerging_degradation": 300,  # Clear pattern; 25+ hours
        "persistent_degradation": 150,  # Sustained; 12+ hours
        "accelerated_deterioration": 60,  # Rapid; 5+ hours
        "chronic_degraded_state": 200,  # Long-term but stable; 16+ hours
        "failure_approach": 20,  # Critical; <2 hours
    }

    # Irreversibility thresholds
    HIGH_SEVERITY_PERSISTENCE_THRESHOLD = 5  # HIGH for N frames = irreversible
    CRITICAL_DEGRADATIONS_THRESHOLD = 3  # 3+ critical relationship failures = irreversible
    DRIFT_ACCELERATION_THRESHOLD = 0.5  # If acceleration this high, window closes fast

    def __init__(self):
        self.previous_window: Optional[InterventionWindow] = None
        self.frame_entered_high_severity: Optional[int] = None
        self.high_severity_persistence = 0

    def compute_window(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        drift_score: float,
        persistence_frames: int,
        temporal_coherence: float,
        finding_confidence: float,
        frame_number: int,
    ) -> DecisionWindow:
        """Compute intervention window for current state.

        Args:
            severity: Current severity level
            degradation_stage: Current degradation stage
            evolution_context: Evolution tracking context
            degradation_map: Relationship degradation analysis
            drift_score: Structural drift [0, 1]
            persistence_frames: Frames at current severity
            temporal_coherence: Signal coherence [0, 1]
            finding_confidence: Confidence in current state
            frame_number: Current frame for trajectory tracking

        Returns:
            DecisionWindow with intervention time estimates
        """
        # Track HIGH severity persistence
        if severity == "HIGH":
            if self.frame_entered_high_severity is None:
                self.frame_entered_high_severity = frame_number
            self.high_severity_persistence = frame_number - self.frame_entered_high_severity
        else:
            self.frame_entered_high_severity = None
            self.high_severity_persistence = 0

        # === COMPUTE BASE WINDOW FROM STAGE ===
        base_window_frames = self.STAGE_WINDOWS.get(degradation_stage, 100)

        # === APPLY VELOCITY COMPRESSION ===
        # Fast-moving systems close windows faster
        velocity_compression = self._compute_velocity_compression(
            evolution_context=evolution_context,
            drift_score=drift_score,
        )
        adjusted_window_frames = int(base_window_frames * (1.0 - velocity_compression))

        # === DETECT IRREVERSIBILITY ===
        is_irreversible = self._detect_irreversibility(
            severity=severity,
            degradation_map=degradation_map,
            persistence_high=self.high_severity_persistence,
            drift_score=drift_score,
            evolution_context=evolution_context,
        )

        if is_irreversible:
            adjusted_window_frames = max(5, adjusted_window_frames // 4)  # Drastically reduced

        # === CLASSIFY WINDOW STATUS ===
        status, narrowing_rate = self._classify_window_status(
            adjusted_window_frames=adjusted_window_frames,
            velocity_compression=velocity_compression,
            is_irreversible=is_irreversible,
        )

        # === COMPUTE PHASE & EFFECTIVENESS ===
        phase, effectiveness = self._compute_phase_and_effectiveness(status)

        # === COMPUTE CONFIDENCE ===
        window_confidence = self._compute_window_confidence(
            temporal_coherence=temporal_coherence,
            finding_confidence=finding_confidence,
            persistence_frames=persistence_frames,
            is_coherent=evolution_context.is_coherent if evolution_context else False,
        )

        # === BUILD INTERVENTION WINDOW ===
        estimated_hours = adjusted_window_frames * self.FRAME_TO_HOURS if adjusted_window_frames > 0 else 0.0
        recommended_urgency = self._recommend_urgency(status, estimated_hours)

        intervention_window = InterventionWindow(
            status=status,
            estimated_hours_remaining=estimated_hours if estimated_hours > 0.1 else None,
            confidence=window_confidence,
            phase=phase,
            effectiveness=effectiveness,
            irreversible_at_frame=self.frame_entered_high_severity if is_irreversible else None,
            narrowing_rate=narrowing_rate,
            recommended_action_urgency=recommended_urgency,
        )

        # === BUILD NARRATIVE ===
        why_closing = self._build_closure_narrative(
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            velocity_compression=velocity_compression,
            is_irreversible=is_irreversible,
            status=status,
        )

        if_wait_one_hour = self._project_window_at_horizon(
            current_frames=adjusted_window_frames,
            hours_ahead=1.0,
        )

        if_wait_next_stage = self._project_impact_next_stage(
            current_stage=degradation_stage,
            current_window_frames=adjusted_window_frames,
        )

        # Track for next frame
        self.previous_window = intervention_window

        return DecisionWindow(
            intervention_window=intervention_window,
            why_closing=why_closing,
            if_wait_one_hour=if_wait_one_hour,
            if_wait_next_stage=if_wait_next_stage,
        )

    def _compute_velocity_compression(
        self,
        evolution_context: Optional[EvolutionContext],
        drift_score: float,
    ) -> float:
        """Compute window compression due to fast movement.

        Returns:
            Compression factor [0, 1]: 0 = no compression, 1 = window closed
        """
        compression = 0.0

        if not evolution_context:
            return compression

        # Accelerating movement compresses window more
        if evolution_context.movement_pattern == "accelerating":
            compression += 0.40

        # High velocity compresses
        if abs(evolution_context.trajectory_velocity) > 0.5:
            compression += 0.20

        # Degrading direction compresses
        if evolution_context.is_degrading:
            compression += 0.15

        # High drift score compresses
        if drift_score > 0.7:
            compression += 0.10

        return min(0.95, compression)  # Never fully closes via velocity alone

    def _detect_irreversibility(
        self,
        severity: SeverityLevel,
        degradation_map: Optional[DegradationMap],
        persistence_high: int,
        drift_score: float,
        evolution_context: Optional[EvolutionContext],
    ) -> bool:
        """Detect when recovery becomes impossible."""
        # Irreversibility condition 1: HIGH severity sustained
        if severity == "HIGH" and persistence_high >= self.HIGH_SEVERITY_PERSISTENCE_THRESHOLD:
            return True

        # Irreversibility condition 2: Critical relationships failing en masse
        if degradation_map and degradation_map.critical_degradation_count >= self.CRITICAL_DEGRADATIONS_THRESHOLD:
            return True

        # Irreversibility condition 3: Drift acceleration extreme
        if evolution_context and evolution_context.trajectory_velocity > self.DRIFT_ACCELERATION_THRESHOLD:
            if drift_score > 0.8:
                return True

        # Irreversibility condition 4: Cascading failure detected
        if degradation_map and degradation_map.is_cascading and degradation_map.critical_degradation_count >= 2:
            return True

        return False

    def _classify_window_status(
        self,
        adjusted_window_frames: int,
        velocity_compression: float,
        is_irreversible: bool,
    ) -> tuple[WindowStatus, str]:
        """Classify window status and narrowing rate."""
        if is_irreversible:
            return "CLOSED", "critical"

        if adjusted_window_frames < 12:  # < 1 hour
            return "CRITICAL", "critical"

        if adjusted_window_frames < 36:  # < 3 hours
            if velocity_compression > 0.5:
                return "NARROWING", "rapid"
            else:
                return "NARROWING", "moderate"

        if adjusted_window_frames < 120:  # < 10 hours
            return "NARROWING", "slow"

        return "OPEN", "slow"

    def _compute_phase_and_effectiveness(self, status: WindowStatus) -> tuple[InterventionPhase, float]:
        """Compute intervention phase and effectiveness curve."""
        phase_map = {
            "OPEN": ("EARLY", 0.9),
            "NARROWING": ("MID", 0.6),
            "CRITICAL": ("LATE", 0.2),
            "CLOSED": ("FINAL", 0.05),
        }
        return phase_map.get(status, ("MID", 0.5))

    def _compute_window_confidence(
        self,
        temporal_coherence: float,
        finding_confidence: float,
        persistence_frames: int,
        is_coherent: bool,
    ) -> float:
        """Compute confidence in window estimate.

        Backed by temporal coherence (signal validity).
        """
        confidence = 0.4  # Base confidence

        # High coherence increases confidence
        if temporal_coherence > 0.7:
            confidence += 0.30

        # High finding confidence increases window confidence
        if finding_confidence > 0.75:
            confidence += 0.20

        # Persistence increases confidence in window
        if persistence_frames >= 5:
            confidence += 0.10

        # Coherence type matters
        if is_coherent:
            confidence += 0.05

        return min(0.95, confidence)

    def _recommend_urgency(self, status: WindowStatus, estimated_hours: Optional[float]) -> str:
        """Recommend action urgency."""
        if status == "CLOSED":
            return "impossible"

        if status == "CRITICAL":
            return "act_now"

        if status == "NARROWING":
            if estimated_hours and estimated_hours < 2:
                return "act_now"
            else:
                return "prepare"

        return "monitor"

    def _build_closure_narrative(
        self,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        velocity_compression: float,
        is_irreversible: bool,
        status: WindowStatus,
    ) -> list[str]:
        """Build list of reasons why window is closing."""
        reasons = []

        if is_irreversible:
            reasons.append("System has crossed irreversibility threshold")

        if evolution_context:
            if evolution_context.movement_pattern == "accelerating":
                reasons.append("Degradation accelerating")
            if evolution_context.is_degrading and evolution_context.trajectory_velocity > 0.3:
                reasons.append("Rapid degradation trajectory")

        if degradation_map:
            if degradation_map.critical_degradation_count >= 2:
                reasons.append(f"{degradation_map.critical_degradation_count} critical relationships failing")
            if degradation_map.is_cascading:
                reasons.append("Cascading failure pattern detected")

        if velocity_compression > 0.5:
            reasons.append("High velocity compressing window")

        if not reasons:
            reasons.append("System transitioning to higher severity state")

        return reasons

    def _project_window_at_horizon(
        self,
        current_frames: int,
        hours_ahead: float,
    ) -> str:
        """Project window size if we wait N hours."""
        frames_ahead = int(hours_ahead / self.FRAME_TO_HOURS)
        remaining_after_wait = max(0, current_frames - frames_ahead)
        hours_remaining = remaining_after_wait * self.FRAME_TO_HOURS

        if remaining_after_wait <= 0:
            return f"If you wait {hours_ahead:.0f}h: window will be CLOSED"
        else:
            return f"If you wait {hours_ahead:.0f}h: window shrinks from {current_frames * self.FRAME_TO_HOURS:.1f}h to {hours_remaining:.1f}h"

    def _project_impact_next_stage(
        self,
        current_stage: DegradationStage,
        current_window_frames: int,
    ) -> str:
        """Project impact if system reaches next stage."""
        stage_progression = [
            "baseline",
            "early_shift",
            "emerging_degradation",
            "persistent_degradation",
            "accelerated_deterioration",
            "chronic_degraded_state",
            "failure_approach",
        ]

        try:
            current_idx = stage_progression.index(current_stage)
        except ValueError:
            return "Cannot project stage impact (unknown stage)"

        if current_idx >= len(stage_progression) - 1:
            return "System at final stage; window nearly closed"

        next_stage = stage_progression[current_idx + 1]
        next_window = self.STAGE_WINDOWS.get(next_stage, 100)
        impact_reduction = (current_window_frames - next_window) / current_window_frames * 100 if current_window_frames > 0 else 0

        return f"Next stage ({next_stage}): window shrinks by {impact_reduction:.0f}%"
