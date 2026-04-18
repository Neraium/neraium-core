"""Trajectory Engine: Map possible paths a system can take next.

Instead of single-point predictions, model the fork in the road:
- What could happen next?
- How likely is each path?
- What structural conditions determine which path?

This is structure-driven, not ML black-box.
Probabilities come from current state + velocity + coherence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
from neraium_core.decision.models import SeverityLevel, DegradationStage, EvolutionContext
from neraium_core.decision.causal_degradation_tracker import DegradationMap


PathName = Literal[
    "stabilization",
    "persistent_degradation",
    "cascade_failure",
    "recovery",
    "chronic_degraded_state",
    "forced_escalation",
]


@dataclass
class TrajectoryPath:
    """One possible path the system could take."""

    name: PathName
    probability: float  # 0-1, based on structural conditions
    likelihood_descriptor: str  # "unlikely" | "possible" | "likely" | "very_likely"
    conditions: list[str]  # Structural conditions that enable this path
    window_impact: str  # How does this path affect intervention window?
    reversibility: bool  # Can the system recover from this path?
    time_to_target: Optional[str]  # "~2 hours", "~1 day", etc.


@dataclass
class TrajectoryBranching:
    """Full branching analysis for a frame."""

    paths: list[TrajectoryPath]
    most_likely_path: TrajectoryPath
    alternative_paths: list[TrajectoryPath]
    convergence_point: Optional[str]  # "All paths lead to HIGH severity unless intervened"
    divergence_reason: str  # Why are there multiple paths? What's uncertain?
    critical_fork: bool  # Is system at a critical branching point?


class TrajectoryEngine:
    """Models possible evolution paths based on structural metrics."""

    # Path definitions: what conditions determine which path
    # More specific to degradation_stage and current metrics

    def __init__(self):
        self.previous_most_likely_path: Optional[PathName] = None
        self.frame_counter = 0

    def branch_trajectories(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        drift_score: float,
        temporal_coherence: float,
        finding_confidence: float,
        persistence_frames: int,
    ) -> TrajectoryBranching:
        """Compute possible paths forward from current state.

        Args:
            severity: Current severity level
            degradation_stage: Current degradation stage
            evolution_context: Evolution tracking context
            degradation_map: Relationship degradation analysis
            drift_score: Structural drift [0, 1]
            temporal_coherence: Signal coherence [0, 1]
            finding_confidence: Confidence in current state
            persistence_frames: Frames at current severity

        Returns:
            TrajectoryBranching with multiple possible paths
        """
        self.frame_counter += 1

        # === GENERATE CANDIDATE PATHS ===
        candidate_paths = self._generate_candidate_paths(
            severity=severity,
            degradation_stage=degradation_stage,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            drift_score=drift_score,
            temporal_coherence=temporal_coherence,
            finding_confidence=finding_confidence,
            persistence_frames=persistence_frames,
        )

        # === RANK PATHS BY PROBABILITY ===
        ranked_paths = sorted(candidate_paths, key=lambda p: p.probability, reverse=True)

        # === IDENTIFY CRITICAL FORK ===
        critical_fork = self._is_critical_fork(ranked_paths)

        # === BUILD BRANCHING ANALYSIS ===
        most_likely = ranked_paths[0] if ranked_paths else self._default_path()
        alternatives = ranked_paths[1:3] if len(ranked_paths) > 1 else []

        divergence_reason = self._explain_divergence(
            ranked_paths, evolution_context, degradation_map
        )
        convergence_point = self._identify_convergence(ranked_paths, degradation_stage)

        self.previous_most_likely_path = most_likely.name

        return TrajectoryBranching(
            paths=ranked_paths,
            most_likely_path=most_likely,
            alternative_paths=alternatives,
            convergence_point=convergence_point,
            divergence_reason=divergence_reason,
            critical_fork=critical_fork,
        )

    def _generate_candidate_paths(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        drift_score: float,
        temporal_coherence: float,
        finding_confidence: float,
        persistence_frames: int,
    ) -> list[TrajectoryPath]:
        """Generate all possible paths from current state."""
        paths: list[TrajectoryPath] = []

        # === PATH 1: STABILIZATION ===
        # Happens when: drift slows, coherence recovers, relationships repair
        stab_prob = self._compute_stabilization_probability(
            drift_score=drift_score,
            temporal_coherence=temporal_coherence,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            degradation_stage=degradation_stage,
        )
        if stab_prob > 0.05:  # Only include if >5% likely
            paths.append(
                TrajectoryPath(
                    name="stabilization",
                    probability=stab_prob,
                    likelihood_descriptor=self._probability_descriptor(stab_prob),
                    conditions=self._stabilization_conditions(
                        evolution_context, degradation_map, degradation_stage
                    ),
                    window_impact="Window widens significantly; low urgency",
                    reversibility=True,
                    time_to_target="~4-24 hours" if stab_prob > 0.3 else None,
                )
            )

        # === PATH 2: PERSISTENT DEGRADATION ===
        # Happens when: current trend continues, slow acceleration
        persist_prob = self._compute_persistent_degradation_probability(
            drift_score=drift_score,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            persistence_frames=persistence_frames,
        )
        if persist_prob > 0.05:
            paths.append(
                TrajectoryPath(
                    name="persistent_degradation",
                    probability=persist_prob,
                    likelihood_descriptor=self._probability_descriptor(persist_prob),
                    conditions=self._persistent_degradation_conditions(
                        evolution_context, drift_score
                    ),
                    window_impact="Window closes slowly; moderate urgency",
                    reversibility=True,
                    time_to_target="~8-48 hours" if persist_prob > 0.4 else None,
                )
            )

        # === PATH 3: CASCADE FAILURE ===
        # Happens when: relationships breaking, acceleration increasing
        cascade_prob = self._compute_cascade_failure_probability(
            degradation_map=degradation_map,
            evolution_context=evolution_context,
            drift_score=drift_score,
        )
        if cascade_prob > 0.05:
            paths.append(
                TrajectoryPath(
                    name="cascade_failure",
                    probability=cascade_prob,
                    likelihood_descriptor=self._probability_descriptor(cascade_prob),
                    conditions=self._cascade_failure_conditions(degradation_map, evolution_context),
                    window_impact="Window closes rapidly; critical urgency",
                    reversibility=False,
                    time_to_target="~1-6 hours" if cascade_prob > 0.2 else None,
                )
            )

        # === PATH 4: RECOVERY ===
        # Happens when: coherence very high, drift low, early stage
        recovery_prob = self._compute_recovery_probability(
            severity=severity,
            degradation_stage=degradation_stage,
            temporal_coherence=temporal_coherence,
            drift_score=drift_score,
        )
        if recovery_prob > 0.05:
            paths.append(
                TrajectoryPath(
                    name="recovery",
                    probability=recovery_prob,
                    likelihood_descriptor=self._probability_descriptor(recovery_prob),
                    conditions=["High coherence", "Low drift", "Early stage"],
                    window_impact="System returns to stable; window widens",
                    reversibility=True,
                    time_to_target="~2-12 hours" if recovery_prob > 0.15 else None,
                )
            )

        # === PATH 5: CHRONIC DEGRADED STATE ===
        # Happens when: persistence high, drift low, adaptation detected
        chronic_prob = self._compute_chronic_degraded_probability(
            degradation_stage=degradation_stage,
            drift_score=drift_score,
            evolution_context=evolution_context,
            persistence_frames=persistence_frames,
        )
        if chronic_prob > 0.05:
            paths.append(
                TrajectoryPath(
                    name="chronic_degraded_state",
                    probability=chronic_prob,
                    likelihood_descriptor=self._probability_descriptor(chronic_prob),
                    conditions=["High persistence", "Slowing drift", "Adaptive behavior"],
                    window_impact="System stabilizes at lower capacity; moderate window",
                    reversibility=True,
                    time_to_target="~24-72 hours",
                )
            )

        # === PATH 6: FORCED ESCALATION ===
        # Happens when: HIGH severity, multiple critical failures
        escalate_prob = self._compute_forced_escalation_probability(
            severity=severity,
            degradation_map=degradation_map,
            drift_score=drift_score,
        )
        if escalate_prob > 0.05:
            paths.append(
                TrajectoryPath(
                    name="forced_escalation",
                    probability=escalate_prob,
                    likelihood_descriptor=self._probability_descriptor(escalate_prob),
                    conditions=["Critical severity", "Multiple relationships failing"],
                    window_impact="Window closes to near-zero; impossible urgency",
                    reversibility=False,
                    time_to_target="<1 hour",
                )
            )

        # Normalize probabilities
        if paths:
            total = sum(p.probability for p in paths)
            for path in paths:
                path.probability = path.probability / total if total > 0 else 1.0 / len(paths)

        return paths if paths else [self._default_path()]

    # === PROBABILITY COMPUTATIONS ===

    def _compute_stabilization_probability(
        self,
        drift_score: float,
        temporal_coherence: float,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        degradation_stage: DegradationStage,
    ) -> float:
        """Probability system stabilizes from here."""
        prob = 0.0

        # Low drift + high coherence = stabilization likely
        if drift_score < 0.3 and temporal_coherence > 0.7:
            prob += 0.35

        # Early stages have higher stabilization chance
        if degradation_stage in {"baseline", "early_shift", "emerging_degradation"}:
            prob += 0.20

        # If not actively degrading, stabilization likely
        if evolution_context and not evolution_context.is_degrading:
            prob += 0.15

        # Few relationships degrading = easier stabilization
        if degradation_map and degradation_map.degradation_count <= 1:
            prob += 0.15

        return min(0.85, prob)

    def _compute_persistent_degradation_probability(
        self,
        drift_score: float,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        persistence_frames: int,
    ) -> float:
        """Probability system stays in current degradation without escalation."""
        prob = 0.0

        # Linear (non-accelerating) movement = persistent path
        if evolution_context and evolution_context.movement_pattern == "linear":
            prob += 0.35

        # Moderate drift = persistent state
        if 0.3 < drift_score < 0.7:
            prob += 0.25

        # Persistent at current level = likely to continue
        if persistence_frames >= 5:
            prob += 0.20

        # Some (not many) relationships degrading
        if degradation_map and 1 <= degradation_map.degradation_count <= 2:
            prob += 0.15

        return min(0.85, prob)

    def _compute_cascade_failure_probability(
        self,
        degradation_map: Optional[DegradationMap],
        evolution_context: Optional[EvolutionContext],
        drift_score: float,
    ) -> float:
        """Probability of cascading failure."""
        prob = 0.0

        # Multiple critical relationships = cascade risk
        if degradation_map and degradation_map.critical_degradation_count >= 2:
            prob += 0.40

        # Accelerating movement = cascade risk
        if evolution_context and evolution_context.movement_pattern == "accelerating":
            prob += 0.30

        # Cascading pattern detected
        if degradation_map and degradation_map.is_cascading:
            prob += 0.25

        # High drift + degrading = cascade likely
        if drift_score > 0.6 and evolution_context and evolution_context.is_degrading:
            prob += 0.20

        return min(0.90, prob)

    def _compute_recovery_probability(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        temporal_coherence: float,
        drift_score: float,
    ) -> float:
        """Probability system recovers to stable."""
        prob = 0.0

        # Early stages have recovery potential
        if degradation_stage in {"baseline", "early_shift"}:
            prob += 0.30

        # LOW severity + high coherence = recovery likely
        if severity == "LOW" and temporal_coherence > 0.8:
            prob += 0.30

        # Very low drift
        if drift_score < 0.2:
            prob += 0.20

        return min(0.70, prob)

    def _compute_chronic_degraded_probability(
        self,
        degradation_stage: DegradationStage,
        drift_score: float,
        evolution_context: Optional[EvolutionContext],
        persistence_frames: int,
    ) -> float:
        """Probability system settles into chronic degraded state."""
        prob = 0.0

        # Chronic stage directly implies this path
        if degradation_stage == "chronic_degraded_state":
            prob += 0.50

        # High persistence + slowing drift = chronic adaptation
        if persistence_frames >= 8 and drift_score < 0.5:
            prob += 0.25

        # Slowing trajectory velocity
        if evolution_context and 0.0 < evolution_context.trajectory_velocity < 0.1:
            prob += 0.15

        return min(0.80, prob)

    def _compute_forced_escalation_probability(
        self,
        severity: SeverityLevel,
        degradation_map: Optional[DegradationMap],
        drift_score: float,
    ) -> float:
        """Probability system escalates to failure."""
        prob = 0.0

        # HIGH severity = escalation risk
        if severity == "HIGH":
            prob += 0.40

        # Many critical failures = forced escalation
        if degradation_map and degradation_map.critical_degradation_count >= 3:
            prob += 0.35

        # Very high drift
        if drift_score > 0.8:
            prob += 0.20

        return min(0.85, prob)

    # === CONDITION EXTRACTION ===

    def _stabilization_conditions(
        self,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        degradation_stage: DegradationStage,
    ) -> list[str]:
        """List conditions that enable stabilization."""
        conditions = []

        if evolution_context:
            if not evolution_context.is_degrading:
                conditions.append("Degradation trajectory stops")
            if evolution_context.temporal_coherence > 0.6:
                conditions.append("Signal coherence sustained")

        if degradation_map and degradation_map.degradation_count <= 1:
            conditions.append("Few relationships degrading")

        if degradation_stage in {"baseline", "early_shift"}:
            conditions.append("Early stage: high recovery potential")

        return conditions or ["Drift slows", "Coherence sustains", "Relationships repair"]

    def _persistent_degradation_conditions(
        self,
        evolution_context: Optional[EvolutionContext],
        drift_score: float,
    ) -> list[str]:
        """Conditions for persistent degradation."""
        conditions = []

        if evolution_context and evolution_context.movement_pattern == "linear":
            conditions.append("Linear degradation pattern continues")

        if 0.3 < drift_score < 0.7:
            conditions.append("Drift remains in moderate range")

        if evolution_context and evolution_context.trajectory_velocity > 0:
            conditions.append("Slight positive velocity maintained")

        return conditions or ["Current trend continues", "No acceleration detected"]

    def _cascade_failure_conditions(
        self,
        degradation_map: Optional[DegradationMap],
        evolution_context: Optional[EvolutionContext],
    ) -> list[str]:
        """Conditions that trigger cascade failure."""
        conditions = []

        if degradation_map and degradation_map.critical_degradation_count >= 2:
            conditions.append(
                f"{degradation_map.critical_degradation_count} critical relationships breaking"
            )

        if evolution_context and evolution_context.movement_pattern == "accelerating":
            conditions.append("Acceleration detected in degradation")

        if degradation_map and degradation_map.is_cascading:
            conditions.append("Cascading failure pattern already active")

        return conditions or ["Multiple relationships fail", "Drift acceleration increases"]

    # === UTILITY METHODS ===

    def _is_critical_fork(self, paths: list[TrajectoryPath]) -> bool:
        """Is system at a critical branching point?"""
        if not paths or len(paths) < 2:
            return False

        # Critical if top two paths have similar probability (uncertain outcome)
        if len(paths) >= 2:
            top_two_diff = paths[0].probability - paths[1].probability
            if top_two_diff < 0.15:  # Within 15% = critical fork
                return True

        # Critical if cascade failure probability high
        cascade = next((p for p in paths if p.name == "cascade_failure"), None)
        if cascade and cascade.probability > 0.25:
            return True

        return False

    def _probability_descriptor(self, prob: float) -> str:
        """Convert probability to descriptor."""
        if prob > 0.70:
            return "very_likely"
        elif prob > 0.45:
            return "likely"
        elif prob > 0.20:
            return "possible"
        else:
            return "unlikely"

    def _explain_divergence(
        self,
        paths: list[TrajectoryPath],
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
    ) -> str:
        """Explain why there are multiple possible paths."""
        if len(paths) < 2:
            return "System trajectory is deterministic from current state."

        if evolution_context and evolution_context.temporal_coherence < 0.5:
            return "Signal noise creates outcome uncertainty."

        if degradation_map and degradation_map.critical_degradation_count >= 2:
            return "Multiple relationship failures create multiple possible progressions."

        if evolution_context and evolution_context.movement_pattern == "oscillating":
            return "Oscillating behavior creates unstable outcome branches."

        return "Current structural state admits multiple progression paths."

    def _identify_convergence(
        self,
        paths: list[TrajectoryPath],
        degradation_stage: DegradationStage,
    ) -> Optional[str]:
        """Identify if all paths eventually converge to same outcome."""
        reversible_paths = [p for p in paths if p.reversibility]
        irreversible_paths = [p for p in paths if not p.reversibility]

        # If some paths are irreversible, convergence point is failure
        if irreversible_paths and len(paths) > 1:
            return "All non-intervention paths converge to failure unless action taken."

        if degradation_stage == "failure_approach":
            return "All paths lead to critical failure unless immediate action."

        return None

    def _default_path(self) -> TrajectoryPath:
        """Fallback path when no candidates generated."""
        return TrajectoryPath(
            name="persistent_degradation",
            probability=1.0,
            likelihood_descriptor="likely",
            conditions=["Current state continued"],
            window_impact="Window narrows normally",
            reversibility=True,
            time_to_target=None,
        )
