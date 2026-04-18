"""Evolution decision resolver: Apply evolution signals to resolve final decisions.

Combines evolution evidence (trajectory, coherence, causal degradation) with
threshold-based severity to produce evolution-informed decisions with confidence boosts.

This is where evolution signals gain decision authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from neraium_core.decision.models import SeverityLevel, EvolutionContext
from neraium_core.decision.causal_degradation_tracker import DegradationMap
from neraium_core.decision.evolution_narrative_builder import EvolutionNarrative


@dataclass
class ConfidenceBreakdown:
    """Component breakdown of final confidence score."""

    base_confidence: float
    trajectory_boost: float = 0.0
    coherence_boost: float = 0.0
    causal_boost: float = 0.0
    model_alignment_boost: float = 0.0
    total_boost: float = 0.0


@dataclass
class EvolutionDecisionResolution:
    """Evolution-informed decision with full justification."""

    severity_from_threshold: SeverityLevel
    severity_from_evolution: SeverityLevel
    final_severity: SeverityLevel
    confidence_breakdown: ConfidenceBreakdown
    evolution_confidence: float  # Separate track of evolution-driven confidence
    override_applied: bool
    override_reason: Optional[str]
    evolution_evidence: list[str]  # Explicit evidence for decision
    recommendation_confidence_boost: float  # How much to boost action confidence


class EvolutionDecisionResolver:
    """Resolves decisions incorporating evolution signals with authority."""

    def __init__(self):
        self.previous_evolution_confidence: Optional[float] = None

    def resolve_with_evolution(
        self,
        threshold_severity: SeverityLevel,
        threshold_confidence: float,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        evolution_narrative: Optional[EvolutionNarrative],
        persistence_frames: int,
    ) -> EvolutionDecisionResolution:
        """Resolve final severity and confidence using evolution signals.

        Args:
            threshold_severity: Severity from threshold logic
            threshold_confidence: Confidence from threshold/statistical logic
            evolution_context: Evolution tracking context
            degradation_map: Relationship degradation analysis
            evolution_narrative: Narrative explaining evolution
            persistence_frames: Frames at current severity

        Returns:
            EvolutionDecisionResolution with final decision and boosts
        """
        confidence_breakdown = ConfidenceBreakdown(base_confidence=threshold_confidence)
        evolution_evidence: list[str] = []
        final_severity = threshold_severity

        # If no evolution context, return unmodified
        if not evolution_context:
            return EvolutionDecisionResolution(
                severity_from_threshold=threshold_severity,
                severity_from_evolution=threshold_severity,
                final_severity=final_severity,
                confidence_breakdown=confidence_breakdown,
                evolution_confidence=0.0,
                override_applied=False,
                override_reason=None,
                evolution_evidence=[],
                recommendation_confidence_boost=0.0,
            )

        # === TRAJECTORY-BASED BOOST ===
        trajectory_boost = self._compute_trajectory_boost(
            trajectory=evolution_context.trajectory_direction,
            velocity=evolution_context.trajectory_velocity,
            movement=evolution_context.movement_pattern,
            persistence=persistence_frames,
        )
        confidence_breakdown.trajectory_boost = trajectory_boost

        if trajectory_boost > 0:
            evolution_evidence.append(
                f"Sustained {evolution_context.movement_pattern} trajectory for {persistence_frames} frames"
            )

        # === COHERENCE-BASED BOOST ===
        coherence_boost = self._compute_coherence_boost(
            temporal_coherence=evolution_context.temporal_coherence,
            is_coherent=evolution_context.is_coherent,
            coherence_type=evolution_context.coherence_type,
        )
        confidence_breakdown.coherence_boost = coherence_boost

        if coherence_boost > 0:
            evolution_evidence.append(f"Strong temporal coherence ({evolution_context.temporal_coherence:.2f})")

        # === CAUSAL BOOST ===
        causal_boost = 0.0
        if degradation_map:
            causal_boost = self._compute_causal_boost(degradation_map)
            confidence_breakdown.causal_boost = causal_boost

            if causal_boost > 0 and degradation_map.critical_degradation_count > 0:
                evolution_evidence.append(
                    f"{degradation_map.critical_degradation_count} critical relationship(s) degrading"
                )

        # === REGIME TRANSITION ALIGNMENT ===
        model_alignment_boost = self._compute_model_alignment_boost(evolution_context)
        confidence_breakdown.model_alignment_boost = model_alignment_boost

        if model_alignment_boost > 0:
            evolution_evidence.append(f"Regime transitions follow known degradation path")

        # === TOTAL EVOLUTION CONFIDENCE ===
        confidence_breakdown.total_boost = (
            trajectory_boost + coherence_boost + causal_boost + model_alignment_boost
        )

        evolution_confidence = min(1.0, threshold_confidence + confidence_breakdown.total_boost)

        # === SEVERITY OVERRIDE LOGIC ===
        override_applied = False
        override_reason: Optional[str] = None
        severity_from_evolution = self._infer_severity_from_evolution(evolution_context, degradation_map)

        # Apply override if evolution suggests higher severity with high confidence
        if (
            severity_from_evolution > threshold_severity  # type: ignore
            and confidence_breakdown.total_boost >= 0.15
            and persistence_frames >= 3
        ):
            final_severity = severity_from_evolution
            override_applied = True
            override_reason = f"Evolution evidence ({confidence_breakdown.total_boost:.2f} boost) suggests {severity_from_evolution}"
            evolution_evidence.insert(0, f"OVERRIDE: Elevated from {threshold_severity} to {final_severity}")

        # Track evolution confidence for next frame
        self.previous_evolution_confidence = evolution_confidence

        recommendation_confidence_boost = min(0.25, confidence_breakdown.total_boost)

        return EvolutionDecisionResolution(
            severity_from_threshold=threshold_severity,
            severity_from_evolution=severity_from_evolution,
            final_severity=final_severity,
            confidence_breakdown=confidence_breakdown,
            evolution_confidence=evolution_confidence,
            override_applied=override_applied,
            override_reason=override_reason,
            evolution_evidence=evolution_evidence,
            recommendation_confidence_boost=recommendation_confidence_boost,
        )

    def _compute_trajectory_boost(
        self,
        trajectory: str,
        velocity: float,
        movement: str,
        persistence: int,
    ) -> float:
        """Compute confidence boost from trajectory signals."""
        boost = 0.0

        # Degrading trajectory with sustained persistence
        if trajectory == "degrading" and persistence >= 5:
            boost += 0.15

        # Accelerating movement adds significant boost
        if movement == "accelerating":
            boost += 0.10

        # Strong velocity adds boost
        if abs(velocity) > 0.3 and persistence >= 3:
            boost += 0.05

        return min(0.3, boost)

    def _compute_coherence_boost(
        self,
        temporal_coherence: float,
        is_coherent: bool,
        coherence_type: str,
    ) -> float:
        """Compute confidence boost from temporal coherence."""
        if temporal_coherence < 0.4:
            return 0.0

        boost = temporal_coherence * 0.15  # Scale coherence to boost range

        # Additional boost for non-noisy signals
        if coherence_type in {"steady", "increasing", "decreasing"}:
            boost += 0.05

        return min(0.25, boost)

    def _compute_causal_boost(self, degradation_map: DegradationMap) -> float:
        """Compute confidence boost from causal degradation evidence."""
        boost = 0.0

        if not degradation_map.degradations_active:
            return 0.0

        # Base boost from any degradation
        boost += 0.05

        # Additional boost for critical degradations
        critical_count = degradation_map.critical_degradation_count
        if critical_count >= 1:
            boost += 0.05 * min(critical_count, 2)

        # Boost for cascading pattern
        if degradation_map.is_cascading:
            boost += 0.05

        # Boost for root cause clarity
        if degradation_map.root_cause_candidate:
            boost += 0.05

        return min(0.25, boost)

    def _compute_model_alignment_boost(self, evolution_context: EvolutionContext) -> float:
        """Compute confidence boost if evolution matches expected patterns."""
        boost = 0.0

        # Multiple transitions indicate state machine progression
        if evolution_context.regime_transitions_count >= 2:
            boost += 0.05

        # Long regime persistence at higher severity indicates committed state
        if evolution_context.regime_persistence >= 8:
            boost += 0.05

        # Regime sequence following typical degradation path
        if len(evolution_context.regime_sequence) >= 2:
            # Simple heuristic: LOW→MODERATE→ELEVATED→HIGH is "expected path"
            recent_pair = (evolution_context.regime_sequence[-2], evolution_context.current_regime)
            expected_pairs = [
                ("STABLE", "WATCH"),
                ("WATCH", "ALERT"),
                ("ALERT", "CRITICAL"),
                ("WATCH", "WATCH"),
                ("ALERT", "ALERT"),
            ]
            if recent_pair in expected_pairs:
                boost += 0.05

        return min(0.15, boost)

    def _infer_severity_from_evolution(
        self,
        evolution_context: EvolutionContext,
        degradation_map: Optional[DegradationMap],
    ) -> SeverityLevel:
        """Infer what severity the evolution signals suggest."""
        # Start with regime mapping
        regime_severity_map = {
            "STABLE": "LOW",
            "WATCH": "MODERATE",
            "ALERT": "ELEVATED",
            "CRITICAL": "HIGH",
        }

        suggested = regime_severity_map.get(evolution_context.current_regime, "MODERATE")

        # Elevate if degrading and accelerating
        if (
            evolution_context.is_degrading
            and evolution_context.movement_pattern == "accelerating"
        ):
            severity_rank = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}
            current_rank = severity_rank.get(suggested, 1)  # type: ignore
            if current_rank < 3:
                rank_to_severity = {0: "LOW", 1: "MODERATE", 2: "ELEVATED", 3: "HIGH"}
                suggested = rank_to_severity[current_rank + 1]  # type: ignore

        # Further elevate if critical relationships degrading
        if (
            degradation_map
            and degradation_map.critical_degradation_count >= 2
            and suggested in {"MODERATE", "ELEVATED"}
        ):
            suggested = "ELEVATED" if suggested == "MODERATE" else "HIGH"  # type: ignore

        return suggested  # type: ignore

    def format_confidence_explanation(
        self, resolution: EvolutionDecisionResolution
    ) -> str:
        """Generate human-readable explanation of confidence score."""
        parts = [
            f"Base: {resolution.confidence_breakdown.base_confidence:.2f}"
        ]

        if resolution.confidence_breakdown.trajectory_boost > 0:
            parts.append(
                f"+ {resolution.confidence_breakdown.trajectory_boost:.2f} (trajectory)"
            )
        if resolution.confidence_breakdown.coherence_boost > 0:
            parts.append(
                f"+ {resolution.confidence_breakdown.coherence_boost:.2f} (coherence)"
            )
        if resolution.confidence_breakdown.causal_boost > 0:
            parts.append(
                f"+ {resolution.confidence_breakdown.causal_boost:.2f} (causal)"
            )
        if resolution.confidence_breakdown.model_alignment_boost > 0:
            parts.append(
                f"+ {resolution.confidence_breakdown.model_alignment_boost:.2f} (model)"
            )

        parts.append(f"= {resolution.evolution_confidence:.2f}")

        return " ".join(parts)
