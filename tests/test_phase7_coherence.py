"""Phase 7: System Coherence and Output Quality Tests

Comprehensive test suite for decision coherence, normalization, and traceability.
Tests ensure no contradictory outputs, consistent messaging, and reliable behavior
across normal and edge-case scenarios.
"""

import pytest
from neraium_core.decision.models import (
    Decision,
    Finding,
    CausalChain,
    PatternMatch,
    TemporalContext,
    PersistenceState,
)
from neraium_core.decision.coherence import (
    validate_decision_coherence,
    enforce_decision_coherence,
    format_coherence_report,
)
from neraium_core.decision.normalization import (
    normalize_confidence,
    normalize_score,
    normalize_decision_output,
    format_for_operator_display,
    validate_numeric_ranges,
)
from neraium_core.decision.traceability import (
    build_decision_trace,
    format_trace_for_display,
    validate_trace,
)


class TestCoherenceValidation:
    """Test coherence validation rules."""

    def test_coherent_baseline_decision(self):
        """Test a fully coherent baseline decision."""
        decision = Decision(
            finding_confidence=0.7,
            action_confidence=0.6,
            transient_score=0.2,
            suppress=False,
            severity="LOW",
            summary="System stable",
            trajectory="stable",
            degradation_stage="baseline",
            action_horizon="watchlist",
            primary_action="continue_observation",
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert is_valid, f"Baseline should be coherent: {errors}"

    def test_suppress_with_action_contradiction(self):
        """Test that suppress=True + action set is caught."""
        decision = Decision(
            finding_confidence=0.3,
            action_confidence=0.2,
            transient_score=0.8,
            suppress=True,
            severity="LOW",
            summary="Transient event",
            trajectory="stable",
            degradation_stage="baseline",
            action_horizon="watchlist",
            primary_action="inspect_subsystem",  # CONTRADICTION
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert not is_valid, "Should catch suppress + action contradiction"
        assert any("suppress" in str(e).lower() for e in errors)

    def test_high_severity_horizon_alignment(self):
        """Test HIGH severity must have now/soon horizon."""
        decision = Decision(
            finding_confidence=0.95,
            action_confidence=0.9,
            transient_score=0.1,
            suppress=False,
            severity="HIGH",
            summary="Critical failure",
            trajectory="degrading",
            degradation_stage="failure_approach",
            action_horizon="watchlist",  # CONTRADICTION
            primary_action="escalate_to_operations",
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert not is_valid, "Should catch HIGH + watchlist contradiction"

    def test_high_severity_not_suppressed(self):
        """Test HIGH severity cannot be suppressed."""
        decision = Decision(
            finding_confidence=0.95,
            action_confidence=0.9,
            transient_score=0.05,
            suppress=True,  # CONTRADICTION
            severity="HIGH",
            summary="Critical failure",
            trajectory="degrading",
            degradation_stage="failure_approach",
            action_horizon="now",
            primary_action="escalate_to_operations",
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert not is_valid, "Should catch HIGH + suppress contradiction"

    def test_failure_approach_stage_must_be_high(self):
        """Test failure_approach stage implies HIGH severity."""
        decision = Decision(
            finding_confidence=0.7,
            action_confidence=0.6,
            transient_score=0.1,
            suppress=False,
            severity="ELEVATED",  # Should be HIGH for failure_approach
            summary="Approaching failure",
            trajectory="degrading",
            degradation_stage="failure_approach",
            action_horizon="soon",
            primary_action="schedule_inspection",
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert not is_valid, "failure_approach should imply HIGH severity"

    def test_degrading_trajectory_not_low_severity(self):
        """Test degrading trajectory should not have LOW severity."""
        decision = Decision(
            finding_confidence=0.5,
            action_confidence=0.4,
            transient_score=0.3,
            suppress=False,
            severity="LOW",  # CONTRADICTION
            summary="System stable",
            trajectory="degrading",  # Should be MODERATE+ for this trajectory
            degradation_stage="baseline",
            action_horizon="watchlist",
            primary_action="continue_observation",
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert not is_valid, "degrading trajectory should not have LOW severity"

    def test_confidence_bounds_validation(self):
        """Test confidence fields stay in [0, 1]."""
        decision = Decision(
            finding_confidence=1.5,  # OUT OF BOUNDS
            action_confidence=0.6,
            transient_score=0.2,
            suppress=False,
            severity="MODERATE",
            summary="Out of bounds",
            trajectory="stable",
            degradation_stage="baseline",
        )

        is_valid, errors = validate_decision_coherence(decision)
        assert not is_valid, "Should catch out-of-bounds confidence"

    def test_coherence_enforcement_suppress_flag(self):
        """Test enforce_decision_coherence clears actions when suppress=True."""
        decision = Decision(
            finding_confidence=0.3,
            action_confidence=0.2,
            transient_score=0.9,
            suppress=True,
            severity="LOW",
            summary="Transient",
            trajectory="stable",
            degradation_stage="baseline",
            action_horizon="watchlist",
            primary_action="inspect_subsystem",
            secondary_actions=["verify_calibration"],
        )

        corrected, corrections = enforce_decision_coherence(decision)

        assert corrected.primary_action is None, "Should clear primary_action when suppressed"
        assert len(corrected.secondary_actions) == 0, "Should clear secondary_actions when suppressed"
        assert any("suppress" in str(c).lower() for c in corrections)


class TestOutputNormalization:
    """Test numeric output normalization."""

    def test_normalize_confidence(self):
        """Test confidence rounding to 2 decimals."""
        assert normalize_confidence(0.876543) == 0.88
        assert normalize_confidence(0.1) == 0.10
        assert normalize_confidence(0.0) == 0.0
        assert normalize_confidence(1.0) == 1.0
        assert normalize_confidence(-0.1) == 0.0  # Clamp
        assert normalize_confidence(1.1) == 1.0  # Clamp

    def test_normalize_score(self):
        """Test score rounding to 3 decimals."""
        assert normalize_score(1.23456) == 1.235
        assert normalize_score(0.0001) == 0.0
        assert normalize_score(2.999) == 2.999

    def test_decision_output_normalization(self):
        """Test full decision normalization."""
        decision = Decision(
            finding_confidence=0.876543,
            action_confidence=0.654321,
            transient_score=0.123456,
            suppress=False,
            severity="HIGH",
            summary="Test",
            trajectory="degrading",
            degradation_stage="baseline",
            action_horizon="now",
            persistence_frames_at_level=5,
            temporal_confidence_delta=0.123456,
        )

        normalized = normalize_decision_output(decision)

        assert normalized["finding_confidence"] == 0.88
        assert normalized["action_confidence"] == 0.65
        assert normalized["transient_score"] == 0.12
        assert normalized["persistence_frames_at_level"] == 5
        assert normalized["temporal_confidence_delta"] == 0.123

    def test_operator_display_format(self):
        """Test operator-facing format removes internal fields."""
        decision = Decision(
            finding_confidence=0.7,
            action_confidence=0.6,
            transient_score=0.2,
            suppress=False,
            severity="ELEVATED",
            summary="Degradation observed",
            trajectory="degrading",
            degradation_stage="emerging_degradation",
            action_horizon="soon",
            primary_action="schedule_inspection",
            reasons=["Reason 1", "Reason 2"],
        )

        display = format_for_operator_display(decision)

        # Internal fields should not be present
        assert "reasons" not in display
        assert "persistence_state" not in display
        assert "temporal_context" not in display

        # Key fields should be present
        assert display["severity"] == "ELEVATED"
        assert display["summary"] == "Degradation observed"

    def test_numeric_range_validation(self):
        """Test numeric bounds validation."""
        # Valid decision
        valid = Decision(
            finding_confidence=0.7,
            action_confidence=0.6,
            transient_score=0.2,
            suppress=False,
            severity="MODERATE",
            summary="Test",
            trajectory="stable",
            degradation_stage="baseline",
            persistence_frames_at_level=3,
            temporal_confidence_delta=0.1,
        )

        is_valid, errors = validate_numeric_ranges(valid)
        assert is_valid, f"Valid decision should pass: {errors}"

        # Invalid: finding_confidence out of range
        invalid = Decision(
            finding_confidence=1.5,
            action_confidence=0.6,
            transient_score=0.2,
            suppress=False,
            severity="MODERATE",
            summary="Test",
            trajectory="stable",
            degradation_stage="baseline",
        )

        is_valid, errors = validate_numeric_ranges(invalid)
        assert not is_valid, "Should catch out-of-bounds confidence"


class TestDecisionTraceability:
    """Test decision traceability and trace generation."""

    def test_trace_high_severity_primary(self):
        """Test HIGH severity becomes primary factor."""
        decision = Decision(
            finding_confidence=0.95,
            action_confidence=0.85,
            transient_score=0.05,
            suppress=False,
            severity="HIGH",
            summary="Critical",
            trajectory="degrading",
            degradation_stage="failure_approach",
        )

        trace = build_decision_trace(decision)

        assert trace is not None
        assert "HIGH" in trace.primary_factor

    def test_trace_stage_transition_primary(self):
        """Test stage transitions are primary factor."""
        decision = Decision(
            finding_confidence=0.7,
            action_confidence=0.6,
            transient_score=0.2,
            suppress=False,
            severity="ELEVATED",
            summary="Stage changed",
            trajectory="degrading",
            degradation_stage="persistent_degradation",
            stage_transition_event="emerging_degradation → persistent_degradation",
        )

        trace = build_decision_trace(decision)

        assert trace is not None
        assert "Stage change" in trace.primary_factor

    def test_trace_pattern_match_secondary(self):
        """Test pattern matches become secondary factors."""
        pattern = PatternMatch(
            pattern_id="test_001",
            similarity=0.92,
            prior_outcome="failure_progression",
            match_tier="strong",
            outcome_type="failure_progression",
        )

        decision = Decision(
            finding_confidence=0.7,
            action_confidence=0.75,
            transient_score=0.1,
            suppress=False,
            severity="ELEVATED",
            summary="Pattern match",
            trajectory="degrading",
            degradation_stage="persistent_degradation",
            pattern_match=pattern,
            pattern_outcome_type="failure_progression",
            pattern_match_tier="strong",
        )

        trace = build_decision_trace(decision)

        assert trace is not None
        assert len(trace.secondary_factors) > 0

    def test_trace_confidence_rationale(self):
        """Test confidence rationale generation."""
        high_conf = Decision(
            finding_confidence=0.95,
            action_confidence=0.9,
            transient_score=0.1,
            suppress=False,
            severity="HIGH",
            summary="High confidence",
            trajectory="degrading",
            degradation_stage="failure_approach",
        )

        trace = build_decision_trace(high_conf)
        assert trace is not None
        assert "High" in trace.confidence_rationale

        low_conf = Decision(
            finding_confidence=0.2,
            action_confidence=0.25,
            transient_score=0.7,
            suppress=True,
            severity="LOW",
            summary="Low confidence",
            trajectory="stable",
            degradation_stage="baseline",
        )

        trace = build_decision_trace(low_conf)
        assert trace is not None
        assert "Low" in trace.confidence_rationale

    def test_trace_format_for_display(self):
        """Test trace formatting for JSON serialization."""
        decision = Decision(
            finding_confidence=0.8,
            action_confidence=0.75,
            transient_score=0.1,
            suppress=False,
            severity="ELEVATED",
            summary="Test",
            trajectory="degrading",
            degradation_stage="emerging_degradation",
        )

        trace = build_decision_trace(decision)
        display = format_trace_for_display(trace)

        assert "decision_trace" in display
        assert "primary_factor" in display["decision_trace"]
        assert "secondary_factors" in display["decision_trace"]
        assert "confidence_rationale" in display["decision_trace"]

    def test_trace_validation(self):
        """Test trace field length validation."""
        decision = Decision(
            finding_confidence=0.8,
            action_confidence=0.75,
            transient_score=0.1,
            suppress=False,
            severity="ELEVATED",
            summary="Test",
            trajectory="stable",
            degradation_stage="baseline",
        )

        trace = build_decision_trace(decision)
        is_valid, errors = validate_trace(trace)

        assert is_valid, f"Generated trace should be valid: {errors}"
        if trace:
            assert len(trace.primary_factor) <= 60
            for factor in trace.secondary_factors:
                assert len(factor) <= 40


class TestEdgeCases:
    """Test edge cases: oscillation, spikes, chronic runs, acceleration."""

    def test_rapid_oscillation_suppression(self):
        """Test decision consistency across rapid severity oscillation."""
        # Simulate frames: ELEVATED → MODERATE → ELEVATED → MODERATE
        decisions = []
        for i, severity in enumerate(["ELEVATED", "MODERATE", "ELEVATED", "MODERATE"]):
            decision = Decision(
                finding_confidence=0.5 + i * 0.05,
                action_confidence=0.45,
                transient_score=0.4,
                suppress=False,
                severity=severity,
                summary=f"Frame {i}: {severity}",
                trajectory="stable",
                degradation_stage="baseline",
                persistence_frames_at_level=1,
            )
            decisions.append(decision)

        # All decisions should be coherent
        for decision in decisions:
            is_valid, errors = validate_decision_coherence(decision)
            assert is_valid, f"Decision {decision.summary} incoherent: {errors}"

    def test_short_lived_spike(self):
        """Test spike that escalates then resolves."""
        decisions = []
        severities = ["LOW", "LOW", "HIGH", "ELEVATED", "MODERATE", "LOW"]

        for i, severity in enumerate(severities):
            horizon = "now" if severity == "HIGH" else ("soon" if severity == "ELEVATED" else "watchlist")
            decision = Decision(
                finding_confidence=0.3 if severity == "LOW" else (0.95 if severity == "HIGH" else 0.7),
                action_confidence=0.2 if severity == "LOW" else (0.9 if severity == "HIGH" else 0.6),
                transient_score=0.8 if i == 2 else 0.3,  # Transient at spike
                suppress=severity == "LOW",
                severity=severity,
                summary=f"Frame {i}: {severity}",
                trajectory="stable",
                degradation_stage="baseline",
                action_horizon=horizon,
                primary_action="escalate_to_operations" if severity == "HIGH" else None,
            )
            decisions.append(decision)

        # All should be coherent
        for i, decision in enumerate(decisions):
            is_valid, errors = validate_decision_coherence(decision)
            assert is_valid, f"Spike test frame {i} incoherent: {errors}"

    def test_long_chronic_degradation(self):
        """Test 60+ frame chronic degradation stays coherent."""
        for frame_num in range(0, 65):
            if frame_num < 20:
                severity = "MODERATE"
                horizon = "soon"
            elif frame_num < 60:
                severity = "ELEVATED"
                horizon = "soon"
            else:
                severity = "HIGH"
                horizon = "now"

            decision = Decision(
                finding_confidence=0.6 + (frame_num * 0.003),
                action_confidence=0.55 + (frame_num * 0.002),
                transient_score=0.2,
                suppress=False,
                severity=severity,
                summary=f"Chronic frame {frame_num}",
                trajectory="degrading",
                degradation_stage="persistent_degradation" if severity in {"ELEVATED", "HIGH"} else "emerging_degradation",
                action_horizon=horizon,
                primary_action="schedule_inspection" if severity == "ELEVATED" else ("escalate_to_operations" if severity == "HIGH" else None),
                persistence_frames_at_level=min(frame_num, 30),
            )

            is_valid, errors = validate_decision_coherence(decision)
            assert is_valid, f"Chronic test frame {frame_num} incoherent: {errors}"

    def test_near_failure_acceleration(self):
        """Test rapid acceleration toward failure."""
        decisions = []
        stages = [
            "emerging_degradation",
            "persistent_degradation",
            "accelerated_deterioration",
            "failure_approach",
        ]
        severities = ["MODERATE", "ELEVATED", "ELEVATED", "HIGH"]
        horizons = ["soon", "soon", "soon", "now"]

        for i, (stage, severity, horizon) in enumerate(zip(stages, severities, horizons)):
            decision = Decision(
                finding_confidence=0.6 + (i * 0.1),
                action_confidence=0.55 + (i * 0.1),
                transient_score=0.1,
                suppress=False,
                severity=severity,
                summary=f"Acceleration frame {i}: {stage}",
                trajectory="degrading",
                degradation_stage=stage,
                action_horizon=horizon,
                primary_action="schedule_inspection" if horizon == "soon" else "escalate_to_operations",
            )
            decisions.append(decision)

        for i, decision in enumerate(decisions):
            is_valid, errors = validate_decision_coherence(decision)
            assert is_valid, f"Acceleration frame {i} incoherent: {errors}"


class TestConsistencyAcrossPhases:
    """Test that Phase 6 and Phase 7 work together."""

    def test_action_horizon_alignment_with_coherence(self):
        """Test action horizon aligns with coherence rules."""
        # Test: HIGH severity must map to now/soon
        for severity in ["HIGH", "ELEVATED", "MODERATE", "LOW"]:
            for horizon in ["now", "soon", "watchlist"]:
                decision = Decision(
                    finding_confidence=0.7,
                    action_confidence=0.6,
                    transient_score=0.2,
                    suppress=severity == "LOW",
                    severity=severity,
                    summary="Test",
                    trajectory="stable",
                    degradation_stage="baseline",
                    action_horizon=horizon,
                    primary_action="escalate_to_operations" if horizon == "now" else None,
                )

                is_valid, errors = validate_decision_coherence(decision)

                if severity == "HIGH":
                    if horizon not in {"now", "soon"}:
                        assert not is_valid, f"HIGH + {horizon} should be incoherent"
                    else:
                        assert is_valid, f"HIGH + {horizon} should be coherent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
