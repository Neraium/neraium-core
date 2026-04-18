"""Intervention Guidance Engine: Make human decisions smarter before failure occurs.

NOT autonomous control. NOT black-box actuation.
Instead: structured guidance on what actions would actually help.

Output shows:
- What action is most likely to help?
- What action buys the most time?
- What action reduces which risks?
- What happens if nothing changes?

Stays read-only. Makes human operators dramatically smarter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal
from neraium_core.decision.models import SeverityLevel, DegradationStage, EvolutionContext
from neraium_core.decision.causal_degradation_tracker import DegradationMap


ActionType = Literal[
    "reduce_load",
    "increase_monitoring",
    "isolate_subsystem",
    "schedule_maintenance",
    "perform_inspection",
    "increase_cooling",
    "adjust_configuration",
    "verify_sensors",
    "do_nothing",
]


@dataclass
class InterventionEffect:
    """Effect of a specific intervention action."""

    action_type: ActionType
    human_readable_name: str  # "Reduce system load"
    description: str  # "Lower operational stress on pressure-flow coupling"

    # Structural impacts
    expected_drift_change: float  # How much does drift improve? (negative = improvement)
    expected_coherence_change: float  # Does signal clarify? (positive = clearer)
    expected_window_extension: float  # How many hours does this buy? (hours)

    # Path impacts
    stabilization_probability_delta: float  # How much more likely is stabilization?
    cascade_failure_probability_delta: float  # How much does cascade risk drop?

    # Timing
    time_to_effect: str  # "immediate" | "1-2 hours" | "4-8 hours"
    reversibility: bool  # Can we undo this if it doesn't help?

    # Confidence
    confidence: float  # 0-1, how confident in this effect prediction?

    # Why this action
    reasoning: str  # Structural basis for why this helps


@dataclass
class InterventionGuidance:
    """Complete guidance on what actions would help."""

    most_impactful_action: InterventionEffect
    fastest_acting_action: InterventionEffect
    time_buying_action: InterventionEffect

    alternative_actions: list[InterventionEffect]

    do_nothing_projection: str  # "If no action: cascade failure path 27% → 67% in ~4 hours"

    critical_decision_points: list[str]  # "If degradation accelerates further, isolation may become irreversible"

    operator_summary: str  # Plain-language summary for control room display


class InterventionGuidanceEngine:
    """Generates structured guidance for human operator decision-making."""

    def __init__(self):
        self.previous_severity: Optional[SeverityLevel] = None
        self.previous_stage: Optional[DegradationStage] = None

    def generate_guidance(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        drift_score: float,
        temporal_coherence: float,
        persistence_frames: int,
        window_status: str,
        estimated_hours_remaining: Optional[float],
        most_likely_path: str,
        cascade_probability: float,
    ) -> InterventionGuidance:
        """Generate intervention guidance for human operators.

        This is read-only: no actuation, only recommendations.
        Shows what actions would help, backed by structural reasoning.

        Args:
            severity: Current severity
            degradation_stage: Current degradation stage
            evolution_context: Evolution tracking
            degradation_map: Relationship degradation analysis
            drift_score: Current structural drift
            temporal_coherence: Signal quality
            persistence_frames: Frames at current state
            window_status: Current intervention window
            estimated_hours_remaining: Hours before window closes
            most_likely_path: Name of most likely evolution path
            cascade_probability: Probability of cascade failure

        Returns:
            InterventionGuidance with structured action recommendations
        """
        # === GENERATE CANDIDATE ACTIONS ===
        candidate_actions = self._generate_candidate_actions(
            severity=severity,
            degradation_stage=degradation_stage,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            drift_score=drift_score,
            temporal_coherence=temporal_coherence,
            window_status=window_status,
            cascade_probability=cascade_probability,
        )

        # === RANK BY IMPACT ===
        most_impactful = max(candidate_actions, key=lambda a: (
            a.stabilization_probability_delta - a.cascade_failure_probability_delta
        )) if candidate_actions else self._default_action()

        # === RANK BY SPEED ===
        fastest = min(candidate_actions, key=lambda a: (
            0 if a.time_to_effect == "immediate" else
            1 if a.time_to_effect.startswith("1-2") else 2
        )) if candidate_actions else self._default_action()

        # === RANK BY TIME BUYING ===
        time_buyer = max(candidate_actions, key=lambda a: a.expected_window_extension) if candidate_actions else self._default_action()

        # === DO NOTHING PROJECTION ===
        do_nothing_proj = self._project_do_nothing(
            most_likely_path=most_likely_path,
            cascade_probability=cascade_probability,
            hours_remaining=estimated_hours_remaining,
        )

        # === CRITICAL DECISION POINTS ===
        critical_points = self._identify_critical_decision_points(
            severity=severity,
            degradation_stage=degradation_stage,
            window_status=window_status,
            cascade_probability=cascade_probability,
        )

        # === OPERATOR SUMMARY ===
        summary = self._build_operator_summary(
            severity=severity,
            most_impactful=most_impactful,
            window_status=window_status,
            estimated_hours=estimated_hours_remaining,
        )

        return InterventionGuidance(
            most_impactful_action=most_impactful,
            fastest_acting_action=fastest,
            time_buying_action=time_buyer,
            alternative_actions=[a for a in candidate_actions if a not in [most_impactful, fastest, time_buyer]][:2],
            do_nothing_projection=do_nothing_proj,
            critical_decision_points=critical_points,
            operator_summary=summary,
        )

    def _generate_candidate_actions(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        evolution_context: Optional[EvolutionContext],
        degradation_map: Optional[DegradationMap],
        drift_score: float,
        temporal_coherence: float,
        window_status: str,
        cascade_probability: float,
    ) -> list[InterventionEffect]:
        """Generate all viable intervention actions for current state."""
        actions: list[InterventionEffect] = []

        # === ACTION 1: REDUCE LOAD ===
        if drift_score > 0.4 or cascade_probability > 0.2:
            actions.append(
                InterventionEffect(
                    action_type="reduce_load",
                    human_readable_name="Reduce operational load",
                    description="Lower stress on system components to slow degradation",
                    expected_drift_change=-0.15 if drift_score > 0.5 else -0.08,
                    expected_coherence_change=0.05,
                    expected_window_extension=2.0 if drift_score > 0.6 else 1.0,
                    stabilization_probability_delta=0.15,
                    cascade_failure_probability_delta=-0.20,
                    time_to_effect="immediate",
                    reversibility=True,
                    confidence=0.80,
                    reasoning="Reduces mechanical stress on degrading relationships; slows cascade propagation",
                )
            )

        # === ACTION 2: INCREASE MONITORING ===
        if temporal_coherence < 0.6 or window_status == "NARROWING":
            actions.append(
                InterventionEffect(
                    action_type="increase_monitoring",
                    human_readable_name="Increase monitoring frequency",
                    description="Sample more frequently to catch transition points early",
                    expected_drift_change=0.0,
                    expected_coherence_change=0.15,
                    expected_window_extension=0.5,  # Buys time by early warning
                    stabilization_probability_delta=0.05,
                    cascade_failure_probability_delta=-0.10,
                    time_to_effect="immediate",
                    reversibility=True,
                    confidence=0.70,
                    reasoning="Improves signal clarity; catches regime transitions before escalation",
                )
            )

        # === ACTION 3: ISOLATE SUBSYSTEM ===
        if degradation_map and degradation_map.critical_degradation_count >= 2 and cascade_probability > 0.20:
            actions.append(
                InterventionEffect(
                    action_type="isolate_subsystem",
                    human_readable_name="Isolate failing subsystem",
                    description="Separate component from rest of system to prevent cascade",
                    expected_drift_change=-0.25,
                    expected_coherence_change=0.20,
                    expected_window_extension=4.0,
                    stabilization_probability_delta=0.25,
                    cascade_failure_probability_delta=-0.50,
                    time_to_effect="1-2 hours",
                    reversibility=False,  # Can't easily reconnect
                    confidence=0.85,
                    reasoning="Breaks cascade chain; prevents multi-component failure",
                )
            )

        # === ACTION 4: SCHEDULE MAINTENANCE ===
        if degradation_stage in {"persistent_degradation", "accelerated_deterioration"}:
            actions.append(
                InterventionEffect(
                    action_type="schedule_maintenance",
                    human_readable_name="Schedule planned maintenance",
                    description="Restore component to baseline before failure",
                    expected_drift_change=-0.40,
                    expected_coherence_change=0.25,
                    expected_window_extension=48.0,  # Resets system
                    stabilization_probability_delta=0.60,
                    cascade_failure_probability_delta=-0.70,
                    time_to_effect="4-8 hours",
                    reversibility=True,  # Maintenance window
                    confidence=0.90,
                    reasoning="Restores relationships to baseline; eliminates root degradation",
                )
            )

        # === ACTION 5: PERFORM INSPECTION ===
        if degradation_stage == "emerging_degradation" or temporal_coherence < 0.5:
            actions.append(
                InterventionEffect(
                    action_type="perform_inspection",
                    human_readable_name="Perform physical inspection",
                    description="Verify sensor calibration and physical component state",
                    expected_drift_change=-0.10,
                    expected_coherence_change=0.20,
                    expected_window_extension=2.0,
                    stabilization_probability_delta=0.10,
                    cascade_failure_probability_delta=-0.05,
                    time_to_effect="1-2 hours",
                    reversibility=True,
                    confidence=0.65,
                    reasoning="Clears signal ambiguity; confirms degradation is real (not sensor artifact)",
                )
            )

        # === ACTION 6: INCREASE COOLING ===
        if degradation_map and any("temperature" in str(d.signal_pair).lower() for d in degradation_map.degradations_active):
            actions.append(
                InterventionEffect(
                    action_type="increase_cooling",
                    human_readable_name="Increase cooling capacity",
                    description="Lower operating temperature to slow thermal degradation",
                    expected_drift_change=-0.12,
                    expected_coherence_change=0.08,
                    expected_window_extension=3.0,
                    stabilization_probability_delta=0.20,
                    cascade_failure_probability_delta=-0.15,
                    time_to_effect="immediate",
                    reversibility=True,
                    confidence=0.75,
                    reasoning="Temperature is primary driver of this degradation path",
                )
            )

        # === ACTION 7: ADJUST CONFIGURATION ===
        if degradation_stage == "early_shift":
            actions.append(
                InterventionEffect(
                    action_type="adjust_configuration",
                    human_readable_name="Adjust operational configuration",
                    description="Change control parameters to avoid degradation trigger conditions",
                    expected_drift_change=-0.08,
                    expected_coherence_change=0.10,
                    expected_window_extension=6.0,
                    stabilization_probability_delta=0.30,
                    cascade_failure_probability_delta=-0.10,
                    time_to_effect="immediate",
                    reversibility=True,
                    confidence=0.60,
                    reasoning="Early stage allows parameter tuning; avoids escalation path",
                )
            )

        return actions

    def _project_do_nothing(
        self,
        most_likely_path: str,
        cascade_probability: float,
        hours_remaining: Optional[float],
    ) -> str:
        """Project what happens if no action is taken."""
        if most_likely_path == "cascade_failure":
            hours = f"~{int(hours_remaining)}" if hours_remaining else "1-2"
            return f"If no action: cascade failure becomes dominant (now {cascade_probability*100:.0f}%) within {hours} hours"

        elif most_likely_path == "persistent_degradation":
            hours = f"~{int(hours_remaining*2)}" if hours_remaining else "8-24"
            return f"If no action: system remains in degraded state, moving toward CRITICAL within {hours} hours"

        elif most_likely_path == "stabilization":
            return "If no action: system may stabilize naturally; monitor for 2-4 hours before confirming"

        else:
            return f"If no action: {most_likely_path} path probability continues to increase"

    def _identify_critical_decision_points(
        self,
        severity: SeverityLevel,
        degradation_stage: DegradationStage,
        window_status: str,
        cascade_probability: float,
    ) -> list[str]:
        """Identify moments where decisions become irreversible or critical."""
        points = []

        if window_status == "CRITICAL":
            points.append("CRITICAL: Window closing to <1 hour; delay increases irreversibility risk")

        if cascade_probability > 0.40:
            points.append("Cascade failure now probable; isolation may become necessary within 1 hour")

        if severity == "HIGH":
            points.append("HIGH severity sustained; recovery becomes progressively harder")

        if degradation_stage == "failure_approach":
            points.append("System approaching failure; all actions have limited efficacy")

        return points

    def _build_operator_summary(
        self,
        severity: SeverityLevel,
        most_impactful: InterventionEffect,
        window_status: str,
        estimated_hours: Optional[float],
    ) -> str:
        """Plain-language summary for control room display."""
        hours_text = f"~{int(estimated_hours)} hours" if estimated_hours else "hours"

        return (
            f"Severity {severity} | Window {window_status} ({hours_text}). "
            f"Recommended: {most_impactful.human_readable_name.lower()} "
            f"(improves stabilization {most_impactful.stabilization_probability_delta*100:.0f}%, "
            f"takes effect {most_impactful.time_to_effect})."
        )

    def _default_action(self) -> InterventionEffect:
        """Fallback when no actions generated."""
        return InterventionEffect(
            action_type="do_nothing",
            human_readable_name="Continue monitoring",
            description="Maintain current operational posture and surveillance",
            expected_drift_change=0.0,
            expected_coherence_change=0.0,
            expected_window_extension=0.0,
            stabilization_probability_delta=0.0,
            cascade_failure_probability_delta=0.0,
            time_to_effect="N/A",
            reversibility=True,
            confidence=0.5,
            reasoning="Current state does not warrant immediate action; continue observation",
        )

    def format_for_operator_display(self, guidance: InterventionGuidance) -> dict:
        """Format guidance for control room display."""
        return {
            "recommended_action": guidance.most_impactful_action.human_readable_name,
            "action_description": guidance.most_impactful_action.description,
            "action_reasoning": guidance.most_impactful_action.reasoning,
            "expected_effect": {
                "drift_improvement": f"{-guidance.most_impactful_action.expected_drift_change:.2f}",
                "window_extension": f"+{guidance.most_impactful_action.expected_window_extension:.1f} hours",
                "stabilization_boost": f"{guidance.most_impactful_action.stabilization_probability_delta*100:.0f}%",
            },
            "time_to_effect": guidance.most_impactful_action.time_to_effect,
            "confidence": f"{guidance.most_impactful_action.confidence*100:.0f}%",
            "do_nothing_projection": guidance.do_nothing_projection,
            "operator_summary": guidance.operator_summary,
        }
