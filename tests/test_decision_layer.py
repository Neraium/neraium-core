"""Comprehensive test suite for the decision layer.

Tests cover:
1. Stable operation (no alert)
2. Transient spike (suppressed)
3. Persistent degradation (surfaced + recommended)
4. Startup disturbance (known safe transient)
5. Repeated prior-pattern match (learning)

Critical invariants:
- HIGH/ELEVATED severity NEVER suppressed
- Decision layer NEVER modifies raw SII outputs
- All decisions deterministic (same input → same output)
"""

import pytest
from copy import deepcopy
from neraium_core.decision import DecisionEngine, PatternMemory
from neraium_core.decision.models import Decision, Finding, SeverityLevel
from neraium_core.decision.confidence import (
    score_finding_confidence,
    score_action_confidence,
    should_surface_finding,
)
from neraium_core.decision.transient_gating import (
    score_transient_likelihood,
    is_known_safe_transient,
    apply_transient_suppression,
)
from neraium_core.decision.specificity import extract_findings
from neraium_core.decision.causal_chains import build_causal_chain
from neraium_core.decision.pattern_memory import PatternMemory, build_feature_vector
from neraium_core.decision.policy import classify_severity, compute_suppress_flag
from neraium_core.decision.recommendation import recommend_action


# ============================================================================
# FIXTURES: Realistic SII Outputs
# ============================================================================

@pytest.fixture
def sii_stable():
    """Stable operation: nominal baseline."""
    return {
        "timestamp": 1704067200.0,
        "asset_id": "pump_standard_001",
        "state": "STABLE",
        "structural_drift_score": 0.12,
        "relational_instability_score": 0.05,
        "system_phase": "stable",
        "policy_alert": False,
        "policy_watch": False,
        "shock_activity": 0.0,
        "subsystem_instability": 0.02,
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "regime_name": "nominal",
        "regime_distance": 0.1,
        "attribution": {
            "top_drivers": [],
            "driver_scores": {},
        },
        "drift_history": [0.10, 0.11, 0.12],
        "data_quality": {
            "missing_sensor_count": 0,
            "valid_signal_count": 3,
        },
    }


@pytest.fixture
def sii_transient_spike():
    """Transient spike: high shock, low sustained trend."""
    return {
        "timestamp": 1704067330.0,
        "asset_id": "pump_standard_001",
        "state": "WATCH",
        "structural_drift_score": 0.52,
        "relational_instability_score": 0.35,
        "system_phase": "transitional",
        "policy_alert": False,
        "policy_watch": True,
        "shock_activity": 0.8,
        "subsystem_instability": 0.28,
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "regime_name": "nominal",
        "regime_distance": 0.15,
        "attribution": {
            "top_drivers": ["vibration"],
            "driver_scores": {"vibration": 0.72},
        },
        "drift_history": [0.10, 0.11, 0.12, 0.45, 0.52],
        "drift_trend": 0.105,
        "data_quality": {
            "missing_sensor_count": 0,
            "valid_signal_count": 3,
        },
    }


@pytest.fixture
def sii_persistent_degradation():
    """Persistent degradation: sustained rise, low shock, degrading phase."""
    return {
        "timestamp": 1704068400.0,
        "asset_id": "pump_A0_001",
        "state": "WATCH",
        "structural_drift_score": 0.58,
        "relational_instability_score": 0.48,
        "system_phase": "degrading",
        "policy_alert": False,
        "policy_watch": True,
        "shock_activity": 0.15,
        "subsystem_instability": 0.42,
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "regime_name": "nominal",
        "regime_distance": 0.22,
        "attribution": {
            "top_drivers": ["pressure", "vibration"],
            "driver_scores": {
                "pressure": 0.85,
                "vibration": 0.78,
            },
        },
        "drift_history": [0.12, 0.15, 0.18, 0.28, 0.40, 0.50, 0.58],
        "drift_trend": 0.076,
        "data_quality": {
            "missing_sensor_count": 0,
            "valid_signal_count": 3,
        },
    }


@pytest.fixture
def sii_startup_disturbance():
    """Startup disturbance: high shock but low frames, matches known pattern."""
    return {
        "timestamp": 1704067200.0,
        "asset_id": "pump_standard_001",
        "state": "WATCH",
        "structural_drift_score": 0.65,
        "relational_instability_score": 0.40,
        "system_phase": "transitional",
        "policy_alert": False,
        "policy_watch": True,
        "shock_activity": 0.9,
        "subsystem_instability": 0.35,
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "regime_name": "startup",
        "regime_distance": 0.5,
        "attribution": {
            "top_drivers": ["temp"],
            "driver_scores": {"temp": 0.65},
        },
        "drift_history": [0.65],
        "data_quality": {
            "missing_sensor_count": 0,
            "valid_signal_count": 3,
        },
    }


@pytest.fixture
def sii_critical_alert():
    """Critical alert: highest severity (ALERT state + high drift)."""
    return {
        "timestamp": 1704069600.0,
        "asset_id": "pump_critical_001",
        "state": "ALERT",
        "structural_drift_score": 0.85,
        "relational_instability_score": 0.72,
        "system_phase": "reorganization",
        "policy_alert": True,
        "policy_watch": True,
        "shock_activity": 0.05,
        "subsystem_instability": 0.68,
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "regime_name": "degraded",
        "regime_distance": 0.8,
        "attribution": {
            "top_drivers": ["pressure", "vibration", "temp"],
            "driver_scores": {
                "pressure": 0.95,
                "vibration": 0.88,
                "temp": 0.72,
            },
        },
        "drift_history": [0.30, 0.42, 0.55, 0.70, 0.78, 0.85],
        "drift_trend": 0.11,
        "time_to_instability": 0.5,
        "data_quality": {
            "missing_sensor_count": 0,
            "valid_signal_count": 3,
        },
    }


@pytest.fixture
def decision_engine():
    """Fresh decision engine for each test."""
    return DecisionEngine()


# ============================================================================
# TEST SUITE 1: STABLE OPERATION
# ============================================================================

class TestStableOperation:
    """Tests for stable, normal operation."""

    def test_stable_has_low_severity(self, sii_stable):
        """Stable state maps to LOW severity."""
        severity = classify_severity(
            state=sii_stable["state"],
            drift_score=sii_stable["structural_drift_score"],
            relational_instability=sii_stable["relational_instability_score"],
            system_phase=sii_stable["system_phase"],
        )
        assert severity == "LOW"

    def test_stable_is_suppressed(self, decision_engine, sii_stable):
        """Stable state is suppressed (routine)."""
        decision = decision_engine.decide(sii_stable, None)
        assert decision.suppress is True
        assert decision.severity == "LOW"

    def test_stable_no_findings(self, sii_stable):
        """Stable state produces no specific findings."""
        findings = extract_findings(
            sii_output=sii_stable,
            attribution=sii_stable.get("attribution"),
            prev_state=None,
        )
        assert len(findings) == 0

    def test_stable_no_recommendation(self, decision_engine, sii_stable):
        """Stable state produces no action recommendation."""
        decision = decision_engine.decide(sii_stable, None)
        assert decision.recommended_action is None

    def test_stable_low_confidence(self, sii_stable):
        """Stable state has low finding confidence."""
        conf = score_finding_confidence(
            drift_score=sii_stable["structural_drift_score"],
            signal_count=3,
            data_quality_issues=0,
            state=sii_stable["state"],
            relational_instability=sii_stable["relational_instability_score"],
        )
        assert conf < 0.5


# ============================================================================
# TEST SUITE 2: TRANSIENT SPIKE
# ============================================================================

class TestTransientSpike:
    """Tests for temporary, self-resolving spikes."""

    def test_spike_high_transient_score(self, sii_transient_spike):
        """High shock + low regime shift = high transient score."""
        transient = score_transient_likelihood(
            drift_score=sii_transient_spike["structural_drift_score"],
            drift_trend=sii_transient_spike.get("drift_trend", 0.0),
            shock_activity=sii_transient_spike["shock_activity"],
            system_phase=sii_transient_spike["system_phase"],
            state=sii_transient_spike["state"],
            drift_history=sii_transient_spike.get("drift_history"),
        )
        assert transient > 0.7, "Spike with high shock should score as transient"

    def test_spike_suppressed_despite_watch_state(self, decision_engine, sii_transient_spike):
        """Transient spike is suppressed even though state=WATCH."""
        decision = decision_engine.decide(sii_transient_spike, None)
        assert decision.suppress is True, "Transient spike should be suppressed"
        assert decision.transient_score > 0.7
        assert decision.severity == "MODERATE"

    def test_spike_has_findings(self, sii_transient_spike):
        """Spike does produce findings (we know something happened)."""
        findings = extract_findings(
            sii_output=sii_transient_spike,
            attribution=sii_transient_spike.get("attribution"),
            prev_state=None,
        )
        assert len(findings) > 0, "Spike should have findings"

    def test_spike_not_recommended(self, decision_engine, sii_transient_spike):
        """Despite findings, transient spike should not trigger recommendation."""
        decision = decision_engine.decide(sii_transient_spike, None)
        assert decision.recommended_action is None, "Transient should not recommend action"

    def test_spike_reasons_explain_suppression(self, decision_engine, sii_transient_spike):
        """Decision reasons should mention transience."""
        decision = decision_engine.decide(sii_transient_spike, None)
        transient_mentioned = any("transient" in r.lower() for r in decision.reasons)
        assert transient_mentioned, "Reasons should explain transience"


# ============================================================================
# TEST SUITE 3: PERSISTENT DEGRADATION
# ============================================================================

class TestPersistentDegradation:
    """Tests for sustained, actionable degradation."""

    def test_degradation_elevated_severity(self, sii_persistent_degradation):
        """Degrading phase + sustained drift = ELEVATED severity."""
        severity = classify_severity(
            state=sii_persistent_degradation["state"],
            drift_score=sii_persistent_degradation["structural_drift_score"],
            relational_instability=sii_persistent_degradation["relational_instability_score"],
            system_phase=sii_persistent_degradation["system_phase"],
        )
        assert severity in {"ELEVATED", "HIGH"}, "Degradation should be ELEVATED or higher"

    def test_degradation_not_suppressed(self, decision_engine, sii_persistent_degradation):
        """Persistent degradation is NOT suppressed."""
        decision = decision_engine.decide(sii_persistent_degradation, None)
        assert decision.suppress is False, "Persistent degradation must be surfaced"

    def test_degradation_low_transient_score(self, sii_persistent_degradation):
        """Degradation has low transient score (not temporary)."""
        transient = score_transient_likelihood(
            drift_score=sii_persistent_degradation["structural_drift_score"],
            drift_trend=sii_persistent_degradation.get("drift_trend", 0.0),
            shock_activity=sii_persistent_degradation["shock_activity"],
            system_phase=sii_persistent_degradation["system_phase"],
            state=sii_persistent_degradation["state"],
            drift_history=sii_persistent_degradation.get("drift_history"),
        )
        assert transient < 0.4, "Persistent degradation should not be transient"

    def test_degradation_has_findings(self, sii_persistent_degradation):
        """Degradation has multiple specific findings."""
        findings = extract_findings(
            sii_output=sii_persistent_degradation,
            attribution=sii_persistent_degradation.get("attribution"),
            prev_state=None,
        )
        assert len(findings) >= 2, "Degradation should have multiple findings"

    def test_degradation_has_causal_chain(self, sii_persistent_degradation):
        """Degradation has causal chain explanation."""
        chain = build_causal_chain(
            sii_output=sii_persistent_degradation,
            shock_activity=sii_persistent_degradation.get("shock_activity", 0.0),
            subsystem_instability=sii_persistent_degradation.get("subsystem_instability", 0.0),
        )
        assert chain is not None, "Degradation should have causal chain"
        assert len(chain.steps) > 0
        assert chain.root_cause is not None

    def test_degradation_recommends_action(self, decision_engine, sii_persistent_degradation):
        """Persistent degradation triggers recommendation."""
        decision = decision_engine.decide(sii_persistent_degradation, None)
        assert decision.recommended_action is not None, "Degradation should recommend action"
        assert decision.recommended_target is not None

    def test_degradation_high_finding_confidence(self, sii_persistent_degradation):
        """Degradation has high finding confidence."""
        conf = score_finding_confidence(
            drift_score=sii_persistent_degradation["structural_drift_score"],
            signal_count=3,
            data_quality_issues=0,
            state=sii_persistent_degradation["state"],
            relational_instability=sii_persistent_degradation["relational_instability_score"],
        )
        assert conf > 0.7, "Degradation should have high finding confidence"


# ============================================================================
# TEST SUITE 4: STARTUP DISTURBANCE
# ============================================================================

class TestStartupDisturbance:
    """Tests for known-safe startup transients."""

    def test_startup_is_safe_transient(self, sii_startup_disturbance):
        """Startup mode is recognized as safe transient."""
        is_safe = is_known_safe_transient(
            state=sii_startup_disturbance["state"],
            system_phase=sii_startup_disturbance["system_phase"],
            regime_name=sii_startup_disturbance["regime_name"],
            recent_events=["startup_initialization"],
        )
        assert is_safe is True

    def test_startup_suppressed(self, decision_engine, sii_startup_disturbance):
        """Startup disturbance is suppressed even with high drift."""
        decision = decision_engine.decide(sii_startup_disturbance, None)
        assert decision.suppress is True, "Known safe startup should be suppressed"

    def test_startup_no_recommendation(self, decision_engine, sii_startup_disturbance):
        """Startup produces no recommendation."""
        decision = decision_engine.decide(sii_startup_disturbance, None)
        assert decision.recommended_action is None

    def test_startup_reasons_explain_safety(self, decision_engine, sii_startup_disturbance):
        """Decision reasons should explain why startup is safe."""
        decision = decision_engine.decide(sii_startup_disturbance, None)
        safe_mentioned = any("startup" in r.lower() or "safe" in r.lower()
                            for r in decision.reasons)
        assert safe_mentioned, "Reasons should mention startup safety"


# ============================================================================
# TEST SUITE 5: CRITICAL INVARIANTS
# ============================================================================

class TestCriticalInvariants:
    """Tests proving critical safety properties."""

    def test_high_severity_never_suppressed(self, decision_engine, sii_critical_alert):
        """HIGH severity is NEVER suppressed (safety invariant)."""
        decision = decision_engine.decide(sii_critical_alert, None)
        if decision.severity == "HIGH":
            assert decision.suppress is False, "HIGH severity MUST NEVER be suppressed"

    def test_elevated_severity_never_suppressed(self, sii_persistent_degradation):
        """ELEVATED severity is NEVER suppressed."""
        suppress_flag = compute_suppress_flag(
            severity="ELEVATED",
            transient_score=0.9,
            finding_confidence=0.2,
        )
        assert suppress_flag is False, "ELEVATED severity MUST NEVER be suppressed"

    def test_critical_alert_highest_urgency(self, decision_engine, sii_critical_alert):
        """ALERT state triggers highest urgency."""
        decision = decision_engine.decide(sii_critical_alert, None)
        assert decision.severity == "HIGH"
        assert decision.suppress is False
        assert decision.recommended_action is not None

    def test_sii_output_not_modified(self, decision_engine, sii_stable):
        """Decision layer NEVER modifies raw SII output."""
        sii_original = deepcopy(sii_stable)
        decision = decision_engine.decide(sii_stable, None)

        assert sii_stable == sii_original, "SII output was modified!"
        assert set(sii_stable.keys()) == set(sii_original.keys())
        assert sii_stable["structural_drift_score"] == sii_original["structural_drift_score"]
        assert sii_stable["relational_instability_score"] == sii_original["relational_instability_score"]

    def test_decision_is_immutable_copy(self, decision_engine, sii_stable):
        """Decision object is independent of SII output."""
        decision = decision_engine.decide(sii_stable, None)
        decision_dict = decision.to_dict()

        decision_dict["severity"] = "MODIFIED"
        decision2 = decision_engine.decide(sii_stable, None)
        assert decision2.severity == "LOW", "Decision should be independent"


# ============================================================================
# TEST SUITE 6: DETERMINISM
# ============================================================================

class TestDeterminism:
    """Tests proving deterministic behavior."""

    def test_same_input_same_output(self, decision_engine, sii_stable):
        """Same SII input always produces identical decision."""
        decision1 = decision_engine.decide(sii_stable, None)
        decision2 = decision_engine.decide(deepcopy(sii_stable), None)

        assert decision1.severity == decision2.severity
        assert decision1.suppress == decision2.suppress
        assert decision1.finding_confidence == decision2.finding_confidence
        assert decision1.transient_score == decision2.transient_score

    def test_multiple_engines_agree(self, sii_stable):
        """Multiple decision engines produce identical decisions."""
        engine1 = DecisionEngine()
        engine2 = DecisionEngine()

        decision1 = engine1.decide(sii_stable, None)
        decision2 = engine2.decide(deepcopy(sii_stable), None)

        assert decision1.to_dict() == decision2.to_dict()

    def test_order_independent(self, sii_stable, sii_transient_spike, decision_engine):
        """Processing order doesn't affect current decision."""
        decision_engine.decide(sii_stable, None)
        decision1 = decision_engine.decide(sii_transient_spike, None)

        engine2 = DecisionEngine()
        decision_engine2 = engine2.decide(sii_transient_spike, None)

        assert decision1.severity == decision_engine2.severity
        assert decision1.suppress == decision_engine2.suppress


# ============================================================================
# TEST SUITE 7: PATTERN MEMORY (LEARNING)
# ============================================================================

class TestPatternMemory:
    """Tests for pattern matching and historical learning."""

    def test_pattern_addition(self, decision_engine):
        """Can add and retrieve patterns."""
        pattern_features = [0.58, 0.48, 0.15, 0.22, 0.5]
        decision_engine.pattern_memory.add_pattern(
            pattern_id="pump_001:run_42",
            features=pattern_features,
            outcome="escalated_to_failure",
            metadata={"time_to_outcome_hours": 12.0},
        )

        match = decision_engine.pattern_memory.find_match(pattern_features, min_similarity=0.95)
        assert match is not None
        assert match.pattern_id == "pump_001:run_42"
        assert match.prior_outcome == "escalated_to_failure"
        assert match.time_to_outcome_hours == 12.0

    def test_pattern_matching_similarity_threshold(self, decision_engine):
        """Pattern matching respects similarity threshold."""
        decision_engine.pattern_memory.add_pattern(
            pattern_id="pattern_1",
            features=[1.0, 0.0, 0.0, 0.0, 0.0],
            outcome="resolved",
        )

        match_exact = decision_engine.pattern_memory.find_match(
            [1.0, 0.0, 0.0, 0.0, 0.0],
            min_similarity=0.95,
        )
        assert match_exact is not None

        match_dissimilar = decision_engine.pattern_memory.find_match(
            [0.0, 1.0, 1.0, 1.0, 1.0],
            min_similarity=0.95,
        )
        assert match_dissimilar is None

    def test_pattern_match_in_decision(self, decision_engine, sii_persistent_degradation):
        """Pattern match appears in decision object when found."""
        feature_vector = build_feature_vector(
            drift_score=sii_persistent_degradation["structural_drift_score"],
            relational_instability=sii_persistent_degradation["relational_instability_score"],
            shock_activity=sii_persistent_degradation["shock_activity"],
            regime_distance=sii_persistent_degradation["regime_distance"],
            system_phase_encoded=0.7,
        )
        decision_engine.pattern_memory.add_pattern(
            pattern_id="historical_degradation",
            features=feature_vector,
            outcome="required_maintenance",
            metadata={"time_to_outcome_hours": 8.0},
        )

        decision = decision_engine.decide(sii_persistent_degradation, None)
        assert decision.pattern_match is not None
        assert decision.pattern_match.prior_outcome == "required_maintenance"


# ============================================================================
# TEST SUITE 8: CONFIDENCE SCORING
# ============================================================================

class TestConfidenceScoring:
    """Tests for independent finding vs action confidence."""

    def test_finding_confidence_depends_on_drift(self):
        """Finding confidence increases with drift score."""
        conf_low = score_finding_confidence(
            drift_score=0.1,
            signal_count=3,
            data_quality_issues=0,
            state="STABLE",
            relational_instability=0.05,
        )

        conf_high = score_finding_confidence(
            drift_score=0.8,
            signal_count=3,
            data_quality_issues=0,
            state="ALERT",
            relational_instability=0.6,
        )

        assert conf_high > conf_low, "Higher drift should increase finding confidence"

    def test_action_confidence_independent_of_finding(self):
        """Action confidence can be low even with high finding confidence."""
        action_conf = score_action_confidence(
            finding_confidence=0.9,
            causal_chain_strength=0.1,
            pattern_match_confidence=0.1,
            recommendation_clarity=0.2,
        )

        assert action_conf < 0.6, "Should be low action confidence despite high finding confidence"

    def test_action_confidence_requires_causal_support(self):
        """Action confidence increases with causal chain strength."""
        conf_weak = score_action_confidence(
            finding_confidence=0.8,
            causal_chain_strength=0.2,
            pattern_match_confidence=0.0,
            recommendation_clarity=0.8,
        )

        conf_strong = score_action_confidence(
            finding_confidence=0.8,
            causal_chain_strength=0.8,
            pattern_match_confidence=0.0,
            recommendation_clarity=0.8,
        )

        assert conf_strong > conf_weak, "Stronger causal chain should increase action confidence"


# ============================================================================
# TEST SUITE 9: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_no_previous_state(self, decision_engine, sii_stable):
        """First frame (no previous state) handled gracefully."""
        decision = decision_engine.decide(sii_stable, prev_output=None)
        assert decision is not None
        assert isinstance(decision.severity, str)

    def test_zero_sensors(self, decision_engine):
        """Zero sensors handled gracefully."""
        sii_no_sensors = {
            "state": "STABLE",
            "structural_drift_score": 0.0,
            "relational_instability_score": 0.0,
            "system_phase": "stable",
            "shock_activity": 0.0,
            "subsystem_instability": 0.0,
            "sensor_relationships": [],
            "attribution": {"top_drivers": []},
        }
        decision = decision_engine.decide(sii_no_sensors, None)
        assert decision is not None

    def test_extreme_drift_value(self, decision_engine):
        """Extreme drift values clamped safely."""
        sii_extreme = {
            "state": "ALERT",
            "structural_drift_score": 999.0,
            "relational_instability_score": 999.0,
            "system_phase": "reorganization",
            "shock_activity": 0.0,
            "subsystem_instability": 0.9,
            "sensor_relationships": ["temp"],
            "attribution": {"top_drivers": ["temp"]},
        }
        decision = decision_engine.decide(sii_extreme, None)

        assert decision.severity == "HIGH"
        assert isinstance(decision.finding_confidence, float)
        assert 0 <= decision.finding_confidence <= 1

    def test_nan_handling(self, decision_engine):
        """NaN values in SII don't crash (graceful degradation)."""
        sii_with_nan = {
            "state": "STABLE",
            "structural_drift_score": 0.0,
            "relational_instability_score": float('nan'),
            "system_phase": "stable",
            "shock_activity": 0.0,
            "subsystem_instability": 0.0,
            "sensor_relationships": [],
            "attribution": {"top_drivers": []},
        }

        decision = decision_engine.decide(sii_with_nan, None)
        assert decision is not None


# ============================================================================
# TEST SUITE 10: INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_decision_flow_stable_to_degradation(self, decision_engine, sii_stable, sii_persistent_degradation):
        """Full flow: stable → degradation → alert."""
        decision1 = decision_engine.decide(sii_stable, None)
        assert decision1.suppress is True

        decision2 = decision_engine.decide(sii_persistent_degradation, sii_stable)
        assert decision2.suppress is False
        assert decision2.recommended_action is not None

    def test_decision_output_structure(self, decision_engine, sii_stable):
        """Decision output has all required fields."""
        decision = decision_engine.decide(sii_stable, None)
        decision_dict = decision.to_dict()

        required_fields = {
            "finding_confidence",
            "action_confidence",
            "transient_score",
            "suppress",
            "severity",
            "summary",
            "findings",
            "causal_chain",
            "pattern_match",
            "recommended_action",
            "recommended_target",
            "reasons",
            # Phase 3 fields
            "trajectory",
            "temporal_confidence_delta",
            "transition_event",
            "temporal_context",
            "consistency_check_passed",
            "persistence_state",
            "persistence_frames_at_level",
            "is_first_appearance",
            # Phase 4 fields
            "degradation_stage",
            "stage_transition_event",
            "stage_specific_recommendation",
        }

        assert set(decision_dict.keys()) == required_fields

    def test_json_serializable(self, decision_engine, sii_stable):
        """Decision object is JSON-serializable."""
        import json
        decision = decision_engine.decide(sii_stable, None)
        decision_dict = decision.to_dict()

        json_str = json.dumps(decision_dict)
        assert isinstance(json_str, str)

        recovered = json.loads(json_str)
        assert recovered["severity"] == "LOW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
