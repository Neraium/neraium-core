"""Decision layer engine: orchestrates all sub-modules.

Takes SII output and produces a Decision object.
"""

from __future__ import annotations

from typing import Any, Optional
from neraium_core.decision.models import Decision, SeverityLevel, PersistenceState, TemporalContext, TrajectoryLabel, DegradationStage
from neraium_core.decision import confidence as conf_module
from neraium_core.decision import transient_gating
from neraium_core.decision import specificity
from neraium_core.decision import causal_chains
from neraium_core.decision import pattern_memory as pm_module
from neraium_core.decision import recommendation
from neraium_core.decision import policy
from neraium_core.decision import persistence_tracker
from neraium_core.decision.temporal_intelligence import TemporalIntelligence
from neraium_core.decision import degradation_stage
from neraium_core.decision.pattern_outcome_influence import PatternOutcomeInfluencer
from neraium_core.decision.action_horizon import ActionHorizonPolicy, compute_primary_action, compute_secondary_actions
from neraium_core.decision import coherence
from neraium_core.decision import traceability as trace_module
from neraium_core.decision import normalization
from neraium_core.decision.evolution_tracker import EvolutionTracker
from neraium_core.decision.temporal_coherence_scoring import TemporalCoherenceScorer
from neraium_core.decision.models import EvolutionContext
from neraium_core.decision.causal_degradation_tracker import CausalDegradationTracker
from neraium_core.decision.evolution_narrative_builder import EvolutionNarrativeBuilder
from neraium_core.decision.evolution_decision_resolver import (
    EvolutionDecisionResolver,
    EvolutionDecisionResolution,
)
from neraium_core.decision.decision_window_engine import DecisionWindowEngine
from neraium_core.decision.trajectory_engine import TrajectoryEngine
from neraium_core.decision.intervention_guidance_engine import InterventionGuidanceEngine
from neraium_core.decision.outcome_tracker import OutcomeTracker
from neraium_core.decision.engine_optimized import DecisionComputationPipeline


class DecisionEngine:
    """Main decision layer orchestrator."""

    def __init__(self, enable_persistence_tracking: bool = True, enable_outcome_tracking: bool = False):
        self.pattern_memory = pm_module.PatternMemory()
        self.pattern_outcome_influencer = PatternOutcomeInfluencer()
        self.persistence_tracker = persistence_tracker.PersistenceTracker()
        self.temporal_intelligence = TemporalIntelligence()
        self.action_horizon_policy = ActionHorizonPolicy()
        self.evolution_tracker = EvolutionTracker()
        self.coherence_scorer = TemporalCoherenceScorer()
        self.degradation_tracker = CausalDegradationTracker()
        self.narrative_builder = EvolutionNarrativeBuilder()
        self.decision_resolver = EvolutionDecisionResolver()
        self.window_engine = DecisionWindowEngine()
        self.trajectory_engine = TrajectoryEngine()
        self.guidance_engine = InterventionGuidanceEngine()
        self.previous_state: Optional[dict[str, Any]] = None
        self.previous_persistence: Optional[PersistenceState] = None
        self.previous_severity: Optional[SeverityLevel] = None
        self.previous_trajectory: Optional[TrajectoryLabel] = None
        self.previous_degradation_stage: Optional[DegradationStage] = None
        self.recent_events: list[str] = []
        self.enable_persistence_tracking = enable_persistence_tracking
        self.frame_counter = 0
        self.outcome_tracker: Optional[OutcomeTracker] = OutcomeTracker() if enable_outcome_tracking else None
        self._pipeline = DecisionComputationPipeline()

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
        temporal_trajectory: TrajectoryLabel = "stable"
        transition_event: Optional[str] = None
        temporal_context: Optional[TemporalContext] = None
        temporal_confidence_delta = 0.0
        consistency_check_passed = True

        if self.enable_persistence_tracking:
            persistence_state = self.persistence_tracker.update_frame(
                current_severity=self._pipeline.phase_severity_classification(
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

        severity = self._pipeline.phase_severity_classification(
            state=state,
            drift_score=drift_score,
            relational_instability=relational_instability,
            system_phase=system_phase,
            prior_severity=self.previous_severity if self.enable_persistence_tracking else None,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
            persistence_trajectory=trajectory if self.enable_persistence_tracking else 0.0,
        )

        # === FINDING CONFIDENCE ===
        finding_confidence = self._pipeline.phase_finding_confidence(
            drift_score=drift_score,
            signal_count=sensor_count,
            data_quality_issues=data_quality_issues,
            state=state,
            relational_instability=relational_instability,
        )

        # === SPRINT 1: EVOLUTION TRACKING ===
        evolution_trajectory = self.evolution_tracker.update_frame(
            severity=severity,
            drift_score=drift_score,
            relational_instability=relational_instability,
            regime_distance=regime_distance,
            finding_confidence=finding_confidence,
        )

        # === PHASE 3: TEMPORAL CONTEXT UPDATE ===
        temporal_context = self.temporal_intelligence.update_temporal_context(
            severity=severity,  # Will use raw policy classification
            drift_score=drift_score,
            finding_confidence=finding_confidence,
        )
        temporal_trajectory = temporal_context.trajectory
        temporal_confidence_delta = self.temporal_intelligence.compute_confidence_delta(
            finding_confidence
        )

        # === TRANSIENT DETECTION ===
        drift_history = sii_output.get("drift_history", [])
        transient_score, is_safe_transient_flag = self._pipeline.phase_transient_detection(
            drift_score=drift_score,
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

        # === SPRINT 1: TEMPORAL COHERENCE SCORING ===
        coherence_profile = self.coherence_scorer.update_metrics(
            metrics={
                "drift_score": drift_score,
                "relational_instability": relational_instability,
                "finding_confidence": finding_confidence,
            }
        )

        # === SPRINT 2: CAUSAL DEGRADATION ANALYSIS ===
        degradation_map = self.degradation_tracker.analyze_relationships(
            sii_output=sii_output,
            relational_instability=relational_instability,
        )

        # === PATTERN MATCHING (moved earlier for Phase 5) ===
        pattern_match = None
        feature_vector = pm_module.build_feature_vector(
            drift_score=drift_score,
            relational_instability=relational_instability,
            shock_activity=shock_activity,
            regime_distance=regime_distance,
            system_phase_encoded=0.5 if system_phase == "degrading" else 0.2,
        )
        pattern_match = self.pattern_memory.find_match(feature_vector)

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

        # === PHASE 5: PATTERN OUTCOME INFLUENCE ON SUPPRESSION ===
        # Check if pattern outcome suggests self-resolution (only for non-HIGH/ELEVATED)
        pattern_based_suppress = self.pattern_outcome_influencer.adjust_suppression_for_self_resolving(
            pattern_match=pattern_match,
            severity=severity,
            transient_score=transient_score,
        )
        if pattern_based_suppress and severity in {"LOW", "MODERATE"}:
            suppress_flag = True

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
            temporal_context=temporal_context,
        )

        causal_chain_strength = causal_chains.chain_strength(causal_chain)

        # === ACTION CONFIDENCE ===
        pattern_match_conf = pattern_match.confidence if pattern_match else 0.0
        recommendation_clarity = 0.8 if severity == "HIGH" else 0.5

        action_confidence = conf_module.score_action_confidence(
            finding_confidence=finding_confidence,
            causal_chain_strength=causal_chain_strength,
            pattern_match_confidence=pattern_match_conf,
            recommendation_clarity=recommendation_clarity,
        )

        # === PHASE 5: PATTERN OUTCOME INFLUENCE ON ACTION CONFIDENCE ===
        action_confidence = self.pattern_outcome_influencer.adjust_action_confidence(
            base_action_confidence=action_confidence,
            pattern_match=pattern_match,
            severity=severity,
        )

        # === PHASE 3: STATE TRANSITION DETECTION ===
        transition_event = self.temporal_intelligence.detect_state_transition(
            previous_trajectory=self.previous_trajectory,
            current_trajectory=temporal_trajectory,
            previous_severity=self.previous_severity,
            current_severity=severity,
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

            # === PHASE 3: DECISION CONSISTENCY LAYER ===
            if rec and self.previous_persistence:
                frames_since_last = self.frame_counter - self.previous_persistence.last_recommendation_frame

                # Check consistency
                consistency_check_passed = self.temporal_intelligence.should_accept_recommendation_flip(
                    new_action=rec.action,
                    last_action=last_action,
                    frames_since_last=frames_since_last,
                    severity_changed=severity_escalated,
                )

                # Also check normal emission logic
                should_emit = recommendation.should_emit_recommendation(
                    current_action=rec.action,
                    last_action=last_action,
                    frames_since_last=frames_since_last,
                    severity_changed=severity_escalated,
                )

                if not (should_emit and consistency_check_passed):
                    rec = None

        # === PHASE 4: DEGRADATION STAGE CLASSIFICATION ===
        drift_velocity = temporal_context.drift_velocity if temporal_context else 0.0
        instability = float(relational_instability) if relational_instability else 0.0

        current_degradation_stage = degradation_stage.classify_degradation_stage(
            severity=severity,
            temporal_context=temporal_context,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
            drift_velocity=drift_velocity,
            instability=instability,
            finding_confidence=finding_confidence,
        )

        # Detect stage transition (to be populated after reasons are built)
        stage_transition_event: Optional[str] = None

        # === SUMMARY ===
        summary = policy.compute_summary(
            severity=severity,
            state=state,
            primary_finding=primary_finding,
            suppress=suppress_flag,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
        )

        # === PHASE 3: ENHANCE SUMMARY WITH TEMPORAL CONTEXT ===
        summary = self.temporal_intelligence.enhance_summary_for_trajectory(
            summary=summary,
            trajectory=temporal_trajectory,
            confidence_trend=temporal_context.confidence_trend,
        )

        # === PHASE 4: ENHANCE SUMMARY WITH DEGRADATION STAGE ===
        stage_summary = degradation_stage.get_stage_summary(current_degradation_stage, severity)
        if not suppress_flag:
            summary = f"{stage_summary} {summary}" if stage_summary and summary else (stage_summary or summary)

        # Get stage-specific recommendation
        stage_rec = degradation_stage.get_stage_recommendation(current_degradation_stage)
        stage_specific_recommendation = stage_rec.get("description", "")

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

        # === PHASE 4: ADD STAGE TRANSITION TO REASONS ===
        if self.previous_degradation_stage is not None and degradation_stage.should_emit_stage_transition(
            self.previous_degradation_stage, current_degradation_stage
        ):
            stage_transition_event = degradation_stage.format_transition_event(
                self.previous_degradation_stage, current_degradation_stage
            )
            reasons.append(f"Stage transition: {self.previous_degradation_stage} → {current_degradation_stage}")

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

        # === PHASE 5: PATTERN OUTCOME INFLUENCE SUMMARY ===
        pattern_influence_summary = self.pattern_outcome_influencer.compute_pattern_influence_summary(
            pattern_match=pattern_match,
            suppress=suppress_flag,
            action_confidence=action_confidence,
            severity=severity,
        )

        # === PHASE 6: ACTION HORIZON INTELLIGENCE ===
        action_horizon, action_priority_reason = self.action_horizon_policy.compute_action_horizon(
            severity=severity,
            degradation_stage=current_degradation_stage,
            trajectory=temporal_trajectory,
            pattern_outcome_type=pattern_match.outcome_type if pattern_match else None,
            pattern_match_tier=pattern_match.match_tier if pattern_match else None,
            drift_score=drift_score,
            finding_confidence=finding_confidence,
            time_to_instability=time_to_instability,
        )

        primary_action = compute_primary_action(
            horizon=action_horizon,
            severity=severity,
            stage=current_degradation_stage,
        )

        secondary_actions = compute_secondary_actions(
            primary=primary_action,
            severity=severity,
            stage=current_degradation_stage,
            drift_score=drift_score,
        )

        # === APPLY EVOLUTION-INFORMED CONFIDENCE BOOST ===
        # (Before decision creation, so it's included in the final output)
        if "action_confidence_evolution_boost" in locals():
            action_confidence = min(1.0, action_confidence + action_confidence_evolution_boost)

        # === SPRINT 1: BUILD EVOLUTION CONTEXT ===
        primary_coherence = coherence_profile.coherences.get("drift_score")
        evolution_context = EvolutionContext(
            regime_sequence=evolution_trajectory.regime_sequence,
            current_regime=evolution_trajectory.regime_sequence[-1] if evolution_trajectory.regime_sequence else "STABLE",
            regime_persistence=evolution_trajectory.current_regime_persistence,
            regime_transitions_count=evolution_trajectory.regime_transitions_count,
            trajectory_direction=evolution_trajectory.trajectory_direction if evolution_trajectory.is_degrading else "stable",
            trajectory_velocity=evolution_trajectory.trajectory_velocity if hasattr(evolution_trajectory, 'trajectory_velocity') else 0.0,
            movement_pattern=evolution_trajectory.movement_pattern,
            is_degrading=evolution_trajectory.is_degrading,
            regime_distance_trajectory=evolution_trajectory.regime_distances,
            temporal_coherence=coherence_profile.overall_system_coherence,
            is_coherent=coherence_profile.is_coherent_overall,
            coherence_type=coherence_profile.primary_signal_type,
            regime_entry_story=self.evolution_tracker.get_regime_entry_story(),
            trajectory_story=self.evolution_tracker.get_trajectory_story(),
        )

        # === SPRINT 2: BUILD EVOLUTION NARRATIVE ===
        evolution_narrative = self.narrative_builder.build_narrative(
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            severity=severity,
            drift_score=drift_score,
            finding_confidence=finding_confidence,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
        )

        # === SPRINT 4: EVOLUTION-INFORMED DECISION RESOLUTION ===
        evolution_resolution = self.decision_resolver.resolve_with_evolution(
            threshold_severity=severity,
            threshold_confidence=finding_confidence,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            evolution_narrative=evolution_narrative,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
        )

        # Apply evolution-informed severity and boost action confidence
        severity = evolution_resolution.final_severity
        action_confidence_evolution_boost = evolution_resolution.recommendation_confidence_boost

        # === DECISION WINDOW ENGINE: When intervention becomes ineffective ===
        decision_window_result = self.window_engine.compute_window(
            severity=severity,
            degradation_stage=current_degradation_stage,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            drift_score=drift_score,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
            temporal_coherence=coherence_profile.overall_system_coherence,
            finding_confidence=finding_confidence,
            frame_number=self.frame_counter,
        )

        # Format window for decision output
        decision_window_output = {
            "status": decision_window_result.intervention_window.status,
            "estimated_hours_remaining": decision_window_result.intervention_window.estimated_hours_remaining,
            "confidence": round(decision_window_result.intervention_window.confidence, 3),
            "phase": decision_window_result.intervention_window.phase,
            "effectiveness": round(decision_window_result.intervention_window.effectiveness, 2),
            "narrowing_rate": decision_window_result.intervention_window.narrowing_rate,
            "recommended_urgency": decision_window_result.intervention_window.recommended_action_urgency,
            "why_closing": decision_window_result.why_closing,
            "if_wait_one_hour": decision_window_result.if_wait_one_hour,
            "if_wait_next_stage": decision_window_result.if_wait_next_stage,
        }

        # === TRAJECTORY BRANCHING: What could happen next? ===
        trajectory_branching_result = self.trajectory_engine.branch_trajectories(
            severity=severity,
            degradation_stage=current_degradation_stage,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            drift_score=drift_score,
            temporal_coherence=coherence_profile.overall_system_coherence,
            finding_confidence=finding_confidence,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
        )

        # Format trajectory for decision output
        trajectory_output = {
            "most_likely_path": {
                "name": trajectory_branching_result.most_likely_path.name,
                "probability": round(trajectory_branching_result.most_likely_path.probability, 3),
                "likelihood": trajectory_branching_result.most_likely_path.likelihood_descriptor,
                "conditions": trajectory_branching_result.most_likely_path.conditions,
                "window_impact": trajectory_branching_result.most_likely_path.window_impact,
                "reversible": trajectory_branching_result.most_likely_path.reversibility,
            },
            "all_paths": [
                {
                    "name": path.name,
                    "probability": round(path.probability, 3),
                    "likelihood": path.likelihood_descriptor,
                    "conditions": path.conditions,
                    "reversible": path.reversibility,
                    "time_to_target": path.time_to_target,
                }
                for path in trajectory_branching_result.paths
            ],
            "critical_fork": trajectory_branching_result.critical_fork,
            "divergence_reason": trajectory_branching_result.divergence_reason,
            "convergence_point": trajectory_branching_result.convergence_point,
        }

        # === INTERVENTION GUIDANCE: What actions would help? ===
        guidance_result = self.guidance_engine.generate_guidance(
            severity=severity,
            degradation_stage=current_degradation_stage,
            evolution_context=evolution_context,
            degradation_map=degradation_map,
            drift_score=drift_score,
            temporal_coherence=coherence_profile.overall_system_coherence,
            persistence_frames=persistence_frames if self.enable_persistence_tracking else 0,
            window_status=decision_window_result.intervention_window.status,
            estimated_hours_remaining=decision_window_result.intervention_window.estimated_hours_remaining,
            most_likely_path=trajectory_branching_result.most_likely_path.name,
            cascade_probability=next(
                (p.probability for p in trajectory_branching_result.paths if p.name == "cascade_failure"),
                0.0,
            ),
        )

        # Format guidance for decision output
        guidance_output = self.guidance_engine.format_for_operator_display(guidance_result)

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
            trajectory=temporal_trajectory,
            temporal_confidence_delta=temporal_confidence_delta,
            transition_event=transition_event,
            temporal_context=temporal_context,
            consistency_check_passed=consistency_check_passed,
            degradation_stage=current_degradation_stage,
            stage_transition_event=stage_transition_event,
            stage_specific_recommendation=stage_specific_recommendation,
            pattern_outcome_type=pattern_match.outcome_type if pattern_match else None,
            pattern_match_tier=pattern_match.match_tier if pattern_match else None,
            pattern_influence_summary=pattern_influence_summary,
            action_horizon=action_horizon,
            primary_action=primary_action,
            secondary_actions=secondary_actions,
            action_priority_reason=action_priority_reason,
            action_tradeoff_note=None,
            evolution_context=evolution_context,
            decision_window=decision_window_output,
            trajectory_branching=trajectory_output,
            intervention_guidance=guidance_output,
        )

        # === PHASE 7: COHERENCE VALIDATION AND TRACEABILITY ===
        # Validate coherence
        is_coherent, coherence_errors = coherence.validate_decision_coherence(decision)
        decision.coherence_errors = coherence_errors

        # Enforce corrections if incoherent
        if not is_coherent:
            decision, corrections = coherence.enforce_decision_coherence(decision)

        # Build decision trace for lightweight traceability
        trace = trace_module.build_decision_trace(decision)
        decision.decision_trace = trace_module.format_trace_for_display(trace).get("decision_trace")

        # Track state
        self.previous_state = sii_output
        self.previous_persistence = next_persistence
        self.previous_severity = severity
        self.previous_trajectory = temporal_trajectory
        self.previous_degradation_stage = current_degradation_stage

        # Log decision to outcome tracker if enabled
        self._log_decision_to_tracker(decision, sii_output)

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

    def _log_decision_to_tracker(self, decision: Decision, sii_output: dict[str, Any]) -> Optional[str]:
        """Log decision to outcome tracker for later evaluation (if enabled).

        Args:
            decision: The decision object
            sii_output: Original SII output

        Returns:
            decision_id if logged, None if tracking disabled
        """
        if self.outcome_tracker is None:
            return None

        asset_id = sii_output.get("asset_id", "unknown")
        reasoning = "; ".join(decision.reasons[:3]) if decision.reasons else decision.summary

        decision_id = self.outcome_tracker.log_decision(
            asset_id=asset_id,
            frame_number=self.frame_counter,
            severity=decision.severity,
            trajectory=decision.trajectory,
            degradation_stage=decision.degradation_stage,
            reasoning_summary=reasoning,
            action_horizon=decision.action_horizon,
            primary_action=decision.primary_action,
            decision_window_status=decision.decision_window.get("status") if decision.decision_window else None,
            pattern_match_tier=decision.pattern_match_tier,
            predicted_outcome_type=decision.pattern_outcome_type,
            confidence=decision.finding_confidence,
            metadata={
                "frame_number": self.frame_counter,
                "transient_score": decision.transient_score,
                "drift_score": float(sii_output.get("structural_drift_score", 0.0)),
            },
        )

        # Store decision_id in decision object for reference
        if decision.decision_trace is None:
            decision.decision_trace = {}
        if isinstance(decision.decision_trace, dict):
            decision.decision_trace["decision_id"] = decision_id

        return decision_id
