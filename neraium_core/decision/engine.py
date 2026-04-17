"""Decision layer engine: orchestrates all sub-modules.

Takes SII output and produces a Decision object.
"""

from __future__ import annotations

from typing import Any, Optional
from neraium_core.decision.models import Decision, SeverityLevel, PersistenceState
from neraium_core.decision import confidence as conf_module
from neraium_core.decision import transient_gating
from neraium_core.decision import specificity
from neraium_core.decision import causal_chains
from neraium_core.decision import pattern_memory as pm_module
from neraium_core.decision import recommendation
from neraium_core.decision import policy
from neraium_core.decision import persistence_tracker


class DecisionEngine:
    """Main decision layer orchestrator."""

    def __init__(self, enable_persistence_tracking: bool = True):
        self.pattern_memory = pm_module.PatternMemory()
        self.persistence_tracker = persistence_tracker.PersistenceTracker()
        self.previous_state: Optional[dict[str, Any]] = None
        self.previous_persistence: Optional[PersistenceState] = None
        self.previous_severity: Optional[SeverityLevel] = None
        self.recent_events: list[str] = []
        self.enable_persistence_tracking = enable_persistence_tracking
        self.frame_counter = 0

    def decide(
        self,
        sii_output: dict[str, Any],
        prev_output: Optional[dict[str, Any]] = None,
    ) -> Decision:
        """Produce a decision from SII output.

        Args:
            sii_output: Output from structural alignment engine
            prev_output: Previous frame's output for delta computation

        Returns:
            Decision object containing finding, action, and meta-information
        """
        self.frame_counter += 1

        # Extract key metrics
        state = sii_output.get("state", "STABLE")
        drift_score = float(sii_output.get("structural_drift_score", 0.0))
        relational_instability = float(sii_output.get("relational_instability_score", 0.0))
        system_phase = sii_output.get("system_phase", "stable")
        regime_name = sii_output.get("regime_name")
        regime_distance = float(sii_output.get("regime_distance") or 0.5)

        attribution = sii_output.get("attribution", {})
        sensor_count = len(sii_output.get("sensor_relationships", []))
        data_quality = sii_output.get("data_quality", {})
        data_quality_issues = 0
        if isinstance(data_quality, dict):
            data_quality_issues = data_quality.get("missing_sensor_count", 0)

        shock_activity = float(sii_output.get("shock_activity") or 0.0)
        subsystem_instability = float(sii_output.get("subsystem_instability") or 0.0)
        time_to_instability = sii_output.get("time_to_instability")

        # === SEVERITY CLASSIFICATION WITH HYSTERESIS ===
        trajectory = 0.0
        persistence_frames = 0
        is_first_appearance = False

        if self.enable_persistence_tracking:
            # Update persistence tracker
            persistence_state = self.persistence_tracker.update_frame(
                current_severity=policy.classify_severity(
                    state=state,
                    drift_score=drift_score,
                    relational_instability=relational_instability,
                    system_phase=system_phase,
                ) if self.previous_severity else "LOW",
                previous_severity=self.previous_severity,
                confidence=0.5,
                state=state,
                frame_number=self.frame_counter,
            )
            trajectory = self.persistence_tracker.compute_trajectory(persistence_state.level_history)
            persistence_frames = persistence_state.consecutive_frames_at_level.get(
                self.previous_severity or "LOW", 0
            )
            is_first_appearance = persistence_frames <= 1

        severity = policy.classify_severity(
            state=state,
            drift_score=drift_score,
            relational_instability=relational_instability,
            system_phase=system_phase,
            prior_severity=self.previous_severity if self.enable_persistence_tracking else None,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
            persistence_trajectory=trajectory if self.enable_persistence_tracking else 0.0,
        )

        # === FINDING CONFIDENCE ===
        finding_confidence = conf_module.score_finding_confidence(
            drift_score=drift_score,
            signal_count=sensor_count,
            data_quality_issues=data_quality_issues,
            state=state,
            relational_instability=relational_instability,
        )

        # === TRANSIENT DETECTION ===
        drift_history = sii_output.get("drift_history", [])
        transient_score = transient_gating.score_transient_likelihood(
            drift_score=drift_score,
            drift_trend=self._compute_drift_trend(drift_history),
            shock_activity=shock_activity,
            system_phase=system_phase,
            state=state,
            drift_history=drift_history if isinstance(drift_history, list) else None,
        )

        is_safe_transient = transient_gating.is_known_safe_transient(
            state=state,
            system_phase=system_phase,
            regime_name=regime_name,
            recent_events=self.recent_events,
        )

        # === SUPPRESSION LOGIC WITH PERSISTENCE ===
        suppress_flag = policy.compute_suppress_flag(
            severity=severity,
            transient_score=transient_score,
            finding_confidence=finding_confidence,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
            is_first_appearance=is_first_appearance if self.enable_persistence_tracking else False,
            frame_number=self.frame_counter if self.enable_persistence_tracking else 0,
        )

        if is_safe_transient and severity in {"LOW", "MODERATE", "ELEVATED"}:
            suppress_flag = True

        # Final check: HIGH never suppressed
        if severity == "HIGH":
            suppress_flag = False

        # === SPECIFIC FINDINGS ===
        findings_list = specificity.extract_findings(
            sii_output=sii_output,
            attribution=attribution,
            prev_state=self.previous_state,
        )

        primary_finding = "System stable"
        if findings_list:
            primary_finding = findings_list[0].description

        # === CAUSAL CHAIN ===
        causal_chain = causal_chains.build_causal_chain(
            sii_output=sii_output,
            shock_activity=shock_activity,
            subsystem_instability=subsystem_instability,
        )

        causal_chain_strength = causal_chains.chain_strength(causal_chain)

        # === PATTERN MATCHING ===
        pattern_match = None
        feature_vector = pm_module.build_feature_vector(
            drift_score=drift_score,
            relational_instability=relational_instability,
            shock_activity=shock_activity,
            regime_distance=regime_distance,
            system_phase_encoded=0.5 if system_phase == "degrading" else 0.2,
        )
        pattern_match = self.pattern_memory.find_match(feature_vector)

        # === ACTION CONFIDENCE ===
        pattern_match_conf = pattern_match.confidence if pattern_match else 0.0
        recommendation_clarity = 0.8 if severity == "HIGH" else 0.5

        action_confidence = conf_module.score_action_confidence(
            finding_confidence=finding_confidence,
            causal_chain_strength=causal_chain_strength,
            pattern_match_confidence=pattern_match_conf,
            recommendation_clarity=recommendation_clarity,
        )

        # === RECOMMENDATIONS WITH ESCALATION ===
        top_signals = []
        if isinstance(attribution, dict):
            drivers = attribution.get("top_drivers", [])
            if isinstance(drivers, list):
                top_signals = drivers[:3]

        last_action = self.previous_persistence.last_recommendation_action if self.previous_persistence else None
        severity_escalated = severity != self.previous_severity if self.previous_severity else False

        rec = None
        if not is_safe_transient and policy.should_recommend(severity=severity, action_confidence=action_confidence):
            rec = recommendation.recommend_action(
                severity=severity,
                state=state,
                drift_score=drift_score,
                time_to_instability=time_to_instability,
                top_signals=top_signals,
                action_confidence=action_confidence,
                persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
                last_action=last_action,
            )

            # Check if we should emit this recommendation
            if rec and self.previous_persistence:
                frames_since_last = self.frame_counter - self.previous_persistence.last_recommendation_frame
                should_emit = recommendation.should_emit_recommendation(
                    current_action=rec.action,
                    last_action=last_action,
                    frames_since_last=frames_since_last,
                    severity_changed=severity_escalated,
                )
                if not should_emit:
                    rec = None

        # === SUMMARY ===
        summary = policy.compute_summary(
            severity=severity,
            state=state,
            primary_finding=primary_finding,
            suppress=suppress_flag,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
        )

        # === REASONS ===
        reasons = self._build_reasons(
            severity=severity,
            finding_confidence=finding_confidence,
            transient_score=transient_score,
            causal_chain_strength=causal_chain_strength,
            suppress=suppress_flag,
            is_safe_transient=is_safe_transient,
            is_first_appearance=is_first_appearance if self.enable_persistence_tracking else False,
        )

        # === BUILD PERSISTENCE STATE FOR NEXT FRAME ===
        next_persistence = None
        if self.enable_persistence_tracking:
            next_persistence = self.persistence_tracker.update_frame(
                current_severity=severity,
                previous_severity=self.previous_severity,
                confidence=finding_confidence,
                state=state,
                frame_number=self.frame_counter,
            )
            if rec:
                next_persistence.last_recommendation_frame = self.frame_counter
                next_persistence.last_recommendation_action = rec.action
                next_persistence.last_recommendation_id = recommendation.recommendation_hash(
                    rec.action, rec.target, rec.urgency
                )

        # === BUILD DECISION ===
        decision = Decision(
            finding_confidence=finding_confidence,
            action_confidence=action_confidence,
            transient_score=transient_score,
            suppress=suppress_flag,
            severity=severity,
            summary=summary,
            findings=findings_list,
            causal_chain=causal_chain,
            pattern_match=pattern_match,
            recommended_action=rec.action if rec else None,
            recommended_target=rec.target if rec else None,
            reasons=reasons,
            persistence_state=next_persistence,
            persistence_frames_at_level=persistence_frames,
            is_first_appearance=is_first_appearance,
        )

        # Track state
        self.previous_state = sii_output
        self.previous_persistence = next_persistence
        self.previous_severity = severity
        return decision

    def _compute_drift_trend(self, drift_history: Any) -> float:
        """Compute drift trend from history."""
        if not isinstance(drift_history, list) or len(drift_history) < 2:
            return 0.0
        try:
            recent = [float(v) for v in drift_history[-5:]]
            return (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        except (ValueError, TypeError):
            return 0.0

    def _build_reasons(
        self,
        severity: SeverityLevel,
        finding_confidence: float,
        transient_score: float,
        causal_chain_strength: float,
        suppress: bool,
        is_safe_transient: bool,
        is_first_appearance: bool = False,
    ) -> list[str]:
        """Build human-readable reasons for the decision."""
        reasons = []

        if severity == "HIGH":
            reasons.append("HIGH severity: immediate attention warranted")

        if is_first_appearance and severity in {"ELEVATED", "MODERATE"}:
            reasons.append("First appearance at this severity level; confirmation in progress")

        if finding_confidence > 0.8:
            reasons.append("Finding is high-confidence")
        elif finding_confidence < 0.4:
            reasons.append("Finding is low-confidence")

        if transient_score > 0.7:
            reasons.append("Likely transient event (may self-resolve)")
        elif transient_score < 0.3:
            reasons.append("Unlikely to be transient")

        if causal_chain_strength > 0.7:
            reasons.append("Causal chain is well-supported")

        if is_safe_transient:
            reasons.append("Matches known safe transient pattern")

        if suppress:
            reasons.append("Suppressed from operator view (low priority)")

        return reasons if reasons else ["No strong signals; baseline behavior"]
