"""Action Horizon Intelligence: Maps decision inputs to action urgency levels.

Distinguishes between:
- now: immediate attention required
- soon: schedule inspection within near term
- watchlist: increase monitoring cadence

Maps stage + severity + trajectory + pattern outcome into action horizons.
Ensures long-run action stability to avoid unnecessary bouncing.
"""

from __future__ import annotations

from typing import Any, Optional, Literal

ActionHorizon = Literal["now", "soon", "watchlist"]


class ActionHorizonPolicy:
    """Determines action horizon from decision context."""

    def __init__(self):
        # Track action history for stability logic
        self.action_horizon_history: list[ActionHorizon] = []
        self.action_severity_history: list[str] = []
        self.action_stage_history: list[str] = []
        self.frame_counter = 0

    def compute_action_horizon(
        self,
        *,
        severity: str,
        degradation_stage: str,
        trajectory: str,
        pattern_outcome_type: Optional[str] = None,
        pattern_match_tier: Optional[str] = None,
        drift_score: float = 0.0,
        finding_confidence: float = 0.5,
        time_to_instability: Optional[float] = None,
    ) -> tuple[ActionHorizon, str]:
        """
        Map decision inputs to action horizon.

        Returns:
            (action_horizon, reason_for_horizon)
        """
        self.frame_counter += 1

        # Map severity + stage + trajectory to base horizon
        base_horizon = self._map_base_horizon(
            severity=severity,
            degradation_stage=degradation_stage,
            trajectory=trajectory,
        )

        # Apply pattern outcome influence
        pattern_influenced = self._apply_pattern_influence(
            base_horizon=base_horizon,
            severity=severity,
            pattern_outcome_type=pattern_outcome_type,
            pattern_match_tier=pattern_match_tier,
        )

        # Apply urgency pressure from time_to_instability
        with_time_pressure = self._apply_time_pressure(
            current_horizon=pattern_influenced,
            severity=severity,
            time_to_instability=time_to_instability,
            drift_score=drift_score,
        )

        # Apply stability logic to prevent unnecessary bouncing
        final_horizon = self._apply_stability_logic(
            proposed_horizon=with_time_pressure,
            severity=severity,
            degradation_stage=degradation_stage,
        )

        # Build reason
        reason = self._build_horizon_reason(
            horizon=final_horizon,
            severity=severity,
            stage=degradation_stage,
            trajectory=trajectory,
            pattern_outcome=pattern_outcome_type,
            time_to_instability=time_to_instability,
        )

        # Record for stability tracking
        self.action_horizon_history.append(final_horizon)
        self.action_severity_history.append(severity)
        self.action_stage_history.append(degradation_stage)

        return final_horizon, reason

    def _map_base_horizon(
        self,
        severity: str,
        degradation_stage: str,
        trajectory: str,
    ) -> ActionHorizon:
        """Map severity + stage + trajectory to base horizon."""

        # HIGH severity always maps to "now"
        if severity == "HIGH":
            return "now"

        # failure_approach stage always maps to "now" (except LOW severity)
        if degradation_stage == "failure_approach" and severity != "LOW":
            return "now"

        # accelerated_deterioration → "now"
        if degradation_stage == "accelerated_deterioration" and severity in {"ELEVATED", "MODERATE"}:
            return "now"

        # ELEVATED + persistent_degradation → "soon"
        if severity == "ELEVATED":
            if degradation_stage in {"persistent_degradation", "accelerated_deterioration"}:
                return "soon"
            # ELEVATED with degrading trajectory → "soon"
            if trajectory == "degrading":
                return "soon"
            # ELEVATED with early stages → "watchlist"
            if degradation_stage in {"early_shift", "emerging_degradation"}:
                return "watchlist"
            # Default ELEVATED → "soon"
            return "soon"

        # MODERATE + degrading → "soon"
        if severity == "MODERATE":
            if trajectory == "degrading" or degradation_stage in {"persistent_degradation", "chronic_degraded_state"}:
                return "soon"
            # MODERATE with early stages → "watchlist"
            return "watchlist"

        # LOW → "watchlist" (observational)
        return "watchlist"

    def _apply_pattern_influence(
        self,
        base_horizon: ActionHorizon,
        severity: str,
        pattern_outcome_type: Optional[str] = None,
        pattern_match_tier: Optional[str] = None,
    ) -> ActionHorizon:
        """Apply pattern outcome information to horizon."""

        if pattern_outcome_type == "failure_progression" and pattern_match_tier in {"moderate", "strong"}:
            # Strong historical evidence of progression → escalate to "now"
            if base_horizon == "soon":
                return "now"

        if pattern_outcome_type == "persistent_degradation" and severity == "ELEVATED":
            # Known persistent patterns at ELEVATED → keep "soon"
            return "soon"

        if pattern_outcome_type == "self_resolved" and severity in {"MODERATE", "LOW"}:
            # Historical self-resolution → lower to "watchlist"
            if base_horizon == "soon":
                return "watchlist"

        # Default: no pattern-based change
        return base_horizon

    def _apply_time_pressure(
        self,
        current_horizon: ActionHorizon,
        severity: str,
        time_to_instability: Optional[float] = None,
        drift_score: float = 0.0,
    ) -> ActionHorizon:
        """Apply urgency pressure from time-to-instability forecasting."""

        if time_to_instability is None:
            return current_horizon

        # Immediate escalation for HIGH + very short time window
        if severity == "HIGH" and time_to_instability <= 3.0:
            return "now"

        # Escalate ELEVATED from "soon" to "now" if window is very short
        if severity == "ELEVATED" and time_to_instability <= 6.0 and current_horizon == "soon":
            return "now"

        # MODERATE + very rapid drift + short window → "soon"
        if (
            severity == "MODERATE"
            and drift_score > 0.5
            and time_to_instability is not None
            and time_to_instability <= 12.0
            and current_horizon == "watchlist"
        ):
            return "soon"

        return current_horizon

    def _apply_stability_logic(
        self,
        proposed_horizon: ActionHorizon,
        severity: str,
        degradation_stage: str,
    ) -> ActionHorizon:
        """Prevent unnecessary action bouncing between horizons."""

        # Need at least 2 frames of history
        if len(self.action_horizon_history) < 2:
            return proposed_horizon

        # Get last two horizons and severities
        last_horizon = self.action_horizon_history[-1]
        prev_severity = self.action_severity_history[-1] if len(self.action_severity_history) >= 1 else "LOW"

        # No change needed (trivial stability)
        if proposed_horizon == last_horizon:
            return proposed_horizon

        # Material severity change justifies horizon change
        severity_escalated = self._is_severity_escalation(prev_severity, severity)
        severity_changed = prev_severity != severity

        # Material stage change justifies horizon change
        prev_stage = self.action_stage_history[-1] if len(self.action_stage_history) >= 1 else "baseline"
        stage_escalated = self._is_stage_escalation(prev_stage, degradation_stage)

        if severity_escalated or stage_escalated:
            return proposed_horizon

        # If horizon would change downward (less urgent), require stronger evidence
        is_downward = self._is_horizon_downgrade(last_horizon, proposed_horizon)

        if is_downward:
            # Downgrades require either:
            # - explicit severity drop, OR
            # - 3+ frames at the new proposed level
            if not severity_changed:
                consecutive_count = sum(
                    1
                    for h in reversed(self.action_horizon_history[-5:])
                    if h == proposed_horizon
                )
                if consecutive_count < 3:
                    # Stick with previous horizon
                    return last_horizon

        return proposed_horizon

    def _is_severity_escalation(self, prev: str, current: str) -> bool:
        """Check if severity escalated."""
        severity_order = {"LOW": 0, "MODERATE": 1, "ELEVATED": 2, "HIGH": 3}
        return severity_order.get(current, 0) > severity_order.get(prev, 0)

    def _is_stage_escalation(self, prev: str, current: str) -> bool:
        """Check if degradation stage escalated materially."""
        stage_order = {
            "baseline": 0,
            "early_shift": 1,
            "emerging_degradation": 2,
            "persistent_degradation": 3,
            "accelerated_deterioration": 4,
            "chronic_degraded_state": 5,
            "failure_approach": 6,
        }
        return stage_order.get(current, 0) > stage_order.get(prev, 0) + 1

    def _is_horizon_downgrade(self, last: ActionHorizon, proposed: ActionHorizon) -> bool:
        """Check if horizon is being downgraded (less urgent)."""
        hierarchy = {"now": 3, "soon": 2, "watchlist": 1}
        return hierarchy.get(proposed, 0) < hierarchy.get(last, 0)

    def _build_horizon_reason(
        self,
        horizon: ActionHorizon,
        severity: str,
        stage: str,
        trajectory: str,
        pattern_outcome: Optional[str] = None,
        time_to_instability: Optional[float] = None,
    ) -> str:
        """Build explanation for why this horizon was chosen."""

        if horizon == "now":
            # Check time pressure first (highest priority reason)
            if time_to_instability is not None:
                if time_to_instability <= 3.0:
                    return f"Immediate action required: very short time to instability ({time_to_instability:.1f} cycles)"
                if time_to_instability <= 6.0 and severity == "ELEVATED":
                    return f"Immediate action required: {severity} severity with {time_to_instability:.1f} cycles to threshold"

            if severity == "HIGH":
                return "Immediate action favored due to HIGH severity classification"
            if stage == "failure_approach":
                return "Immediate action required: failure_approach stage detected"
            if pattern_outcome == "failure_progression":
                return f"Immediate action required: failure_progression pattern detected"
            return f"Immediate action required: {severity} severity at {stage}"

        if horizon == "soon":
            if severity == "ELEVATED":
                if stage == "persistent_degradation":
                    return "Schedule inspection: ELEVATED severity with persistent degradation pattern"
                if trajectory == "degrading":
                    return "Schedule inspection: ELEVATED severity with degrading trajectory"
            if trajectory == "degrading" and stage in {"persistent_degradation", "accelerated_deterioration"}:
                return f"Schedule inspection: degrading trajectory at {stage}"
            if pattern_outcome == "persistent_degradation":
                return "Schedule inspection: historically persistent degradation pattern"
            if time_to_instability is not None and time_to_instability <= 12.0:
                return f"Schedule inspection: {time_to_instability:.1f} cycles to threshold"
            return f"Schedule inspection: {severity} severity at {stage}"

        # watchlist
        if pattern_outcome == "self_resolved":
            return "Increase monitoring: pattern historically self-resolves"
        if stage == "early_shift":
            return "Increase monitoring: early_shift stage with limited persistence"
        if severity == "MODERATE" and trajectory == "stable":
            return "Increase monitoring cadence: stable MODERATE severity"
        return f"Increase monitoring: {severity} severity at {stage} stage"


def compute_primary_action(
    horizon: ActionHorizon,
    severity: str,
    stage: str,
) -> str:
    """Map horizon + severity + stage to primary recommended action."""

    if horizon == "now":
        if stage == "failure_approach":
            return "escalate_to_operations"
        if severity == "HIGH":
            return "urgent_escalation"
        return "escalate_to_operations"

    if horizon == "soon":
        if stage in {"persistent_degradation", "accelerated_deterioration"}:
            return "schedule_inspection"
        if severity == "ELEVATED":
            return "inspect_subsystem"
        return "schedule_inspection"

    # watchlist
    if stage in {"early_shift", "emerging_degradation"}:
        return "increase_monitoring"
    return "continue_observation"


def compute_secondary_actions(
    primary: str,
    severity: str,
    stage: str,
    drift_score: float = 0.0,
) -> list[str]:
    """Generate secondary actions to support primary action."""

    actions = []

    # Always validate sensors for non-LOW severity
    if severity in {"ELEVATED", "HIGH"}:
        actions.append("validate_sensor_calibration")

    # Sampling cadence
    if severity in {"ELEVATED", "HIGH"}:
        actions.append("increase_sampling_frequency")

    # Subsystem inspection
    if drift_score > 0.4 and stage in {"persistent_degradation", "accelerated_deterioration"}:
        actions.append("inspect_subsystems")

    # Configuration review for structural changes
    if stage in {"early_shift", "regime_shift_observed"}:
        actions.append("review_configuration_changes")

    # Failover prep for failure approach
    if stage == "failure_approach":
        actions.append("prepare_failover_routing")

    return actions[:3]  # Limit to 3 secondary actions
