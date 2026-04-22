"""Phase 4: Degradation Stage Modeling Tests

Validates:
1. Degradation stage classification across 7 stages
2. Stage-specific summaries and recommendations
3. Stage transition events
4. Long-run behavior with message compression
5. Trajectory semantics in degradation context
"""

from __future__ import annotations

import json
from neraium_core.decision import DecisionEngine
from neraium_core.decision.degradation_stage import (
    classify_degradation_stage,
    get_stage_summary,
    get_stage_recommendation,
    DegradationStage,
)
from neraium_core.decision.models import TemporalContext


def test_stage_classifier_baseline():
    """Test baseline stage classification."""
    tc = TemporalContext(
        severity_history=["LOW"],
        drift_scores=[0.1],
        confidence_scores=[0.2],
        trajectory="stable",
    )

    stage = classify_degradation_stage(
        severity="LOW",
        temporal_context=tc,
        persistence_frames=1,
        drift_velocity=0.0,
        instability=0.05,
        finding_confidence=0.2,
    )
    assert stage == "baseline"


def test_stage_classifier_early_shift():
    """Test early_shift stage classification."""
    tc = TemporalContext(
        severity_history=["LOW", "MODERATE"],
        drift_scores=[0.1, 0.35],
        confidence_scores=[0.2, 0.5],
        trajectory="stable",
    )

    stage = classify_degradation_stage(
        severity="MODERATE",
        temporal_context=tc,
        persistence_frames=1,
        drift_velocity=0.1,
        instability=0.3,
        finding_confidence=0.5,
    )
    assert stage == "early_shift"


def test_stage_classifier_emerging_degradation():
    """Test emerging_degradation stage classification."""
    tc = TemporalContext(
        severity_history=["LOW", "MODERATE", "ELEVATED"],
        drift_scores=[0.1, 0.35, 0.5],
        confidence_scores=[0.2, 0.5, 0.65],
        trajectory="degrading",
    )

    stage = classify_degradation_stage(
        severity="ELEVATED",
        temporal_context=tc,
        persistence_frames=2,
        drift_velocity=0.15,
        instability=0.5,
        finding_confidence=0.65,
    )
    assert stage == "emerging_degradation"


def test_stage_classifier_persistent_degradation():
    """Test persistent_degradation stage classification."""
    tc = TemporalContext(
        severity_history=["MODERATE", "ELEVATED", "ELEVATED", "ELEVATED", "ELEVATED"],
        drift_scores=[0.35, 0.5, 0.55, 0.6, 0.65],
        confidence_scores=[0.5, 0.65, 0.68, 0.7, 0.72],
        trajectory="degrading",
    )

    stage = classify_degradation_stage(
        severity="ELEVATED",
        temporal_context=tc,
        persistence_frames=4,
        drift_velocity=0.1,
        instability=0.6,
        finding_confidence=0.72,
    )
    assert stage == "persistent_degradation"


def test_stage_classifier_accelerated_deterioration():
    """Test accelerated_deterioration stage classification."""
    tc = TemporalContext(
        severity_history=["ELEVATED"] * 5,
        drift_scores=[0.6, 0.7, 0.8, 0.9, 1.0],
        confidence_scores=[0.7, 0.75, 0.8, 0.82, 0.84],
        trajectory="degrading",
    )

    stage = classify_degradation_stage(
        severity="ELEVATED",
        temporal_context=tc,
        persistence_frames=5,
        drift_velocity=0.3,  # High drift velocity
        instability=0.8,
        finding_confidence=0.84,
    )
    assert stage == "accelerated_deterioration"


def test_stage_classifier_failure_approach():
    """Test failure_approach stage classification."""
    tc = TemporalContext(
        severity_history=["HIGH"] * 3,
        drift_scores=[1.2, 1.4, 1.6],
        confidence_scores=[0.85, 0.87, 0.9],
        trajectory="degrading",
    )

    stage = classify_degradation_stage(
        severity="HIGH",
        temporal_context=tc,
        persistence_frames=2,
        drift_velocity=0.2,
        instability=0.9,
        finding_confidence=0.9,
    )
    assert stage == "failure_approach"


def test_stage_summaries_are_distinct():
    """Verify each stage has a distinct summary message."""
    stages: list[DegradationStage] = [
        "baseline",
        "early_shift",
        "emerging_degradation",
        "persistent_degradation",
        "accelerated_deterioration",
        "chronic_degraded_state",
        "failure_approach",
    ]

    summaries = [get_stage_summary(stage, "MODERATE") for stage in stages]

    # All summaries should be distinct
    assert len(set(summaries)) == len(summaries), "Stage summaries should be unique"

    # All summaries should be non-empty
    assert all(summaries), "All stage summaries should be non-empty"

    # Summaries should reflect the stage name
    assert "baseline" in summaries[0].lower() or "normal" in summaries[0].lower()
    assert "shift" in summaries[1].lower() or "early" in summaries[1].lower()
    assert "failure" in summaries[-1].lower()


def test_stage_recommendations_vary_by_stage():
    """Verify recommendations vary by degradation stage."""
    stages: list[DegradationStage] = [
        "baseline",
        "early_shift",
        "emerging_degradation",
        "persistent_degradation",
        "accelerated_deterioration",
        "chronic_degraded_state",
        "failure_approach",
    ]

    recommendations = [get_stage_recommendation(stage) for stage in stages]
    actions = [r["action"] for r in recommendations]

    # Actions should vary
    assert len(set(actions)) > 1, "Stage recommendations should not all be the same"

    # Early stages should recommend monitoring
    assert "monitor" in actions[0].lower()

    # Later stages should recommend escalation
    assert "escalate" in actions[4].lower() or "maintenance" in actions[4].lower()
    assert "immediate" in actions[6].lower()


def test_long_run_degradation_with_stage_progression():
    """Test stage progression across 20+ frames of degradation."""
    engine = DecisionEngine(enable_persistence_tracking=True)

    # Simulate degradation progression
    frames = []
    stage_sequence = []

    # Baseline frames (1-5)
    for i in range(5):
        frame = {
            "asset_id": "pump_test_001",
            "state": "STABLE",
            "structural_drift_score": 0.1,
            "relational_instability_score": 0.05,
            "system_phase": "stable",
            "policy_watch": False,
            "shock_activity": 0.0,
            "subsystem_instability": 0.02,
            "sensor_relationships": ["temp", "pressure"],
            "regime_name": "nominal",
            "regime_distance": 0.1,
            "drift_history": [0.08, 0.09, 0.1],
            "attribution": {"top_drivers": [], "driver_scores": {}},
            "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 2},
        }
        decision = engine.decide(frame)
        frames.append(decision)
        stage_sequence.append(decision.degradation_stage)

    # Early shift frames (6-8)
    for i in range(3):
        frame = {
            "asset_id": "pump_test_001",
            "state": "WATCH",
            "structural_drift_score": 0.35,
            "relational_instability_score": 0.25,
            "system_phase": "stable",
            "policy_watch": True,
            "shock_activity": 0.1,
            "subsystem_instability": 0.15,
            "sensor_relationships": ["temp", "pressure"],
            "regime_name": "nominal",
            "regime_distance": 0.2,
            "drift_history": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
            "attribution": {"top_drivers": ["temp"], "driver_scores": {"temp": 0.6}},
            "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 2},
        }
        decision = engine.decide(frame)
        frames.append(decision)
        stage_sequence.append(decision.degradation_stage)

    # Emerging degradation frames (9-15)
    for i in range(7):
        drift = 0.35 + (i * 0.08)
        frame = {
            "asset_id": "pump_test_001",
            "state": "WATCH",
            "structural_drift_score": drift,
            "relational_instability_score": 0.4,
            "system_phase": "degrading",
            "policy_watch": True,
            "shock_activity": 0.05,
            "subsystem_instability": 0.3,
            "sensor_relationships": ["temp", "pressure"],
            "regime_name": "nominal",
            "regime_distance": 0.3,
            "drift_history": list(range(0, int(drift * 10))),
            "attribution": {"top_drivers": ["pressure"], "driver_scores": {"pressure": 0.75}},
            "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 2},
        }
        decision = engine.decide(frame)
        frames.append(decision)
        stage_sequence.append(decision.degradation_stage)

    # Persistent degradation frames (16-25)
    for i in range(10):
        drift = 0.9 + (i * 0.05)
        frame = {
            "asset_id": "pump_test_001",
            "state": "ALERT",
            "structural_drift_score": drift,
            "relational_instability_score": 0.6,
            "system_phase": "degrading",
            "policy_alert": True,
            "shock_activity": 0.02,
            "subsystem_instability": 0.5,
            "sensor_relationships": ["temp", "pressure"],
            "regime_name": "degraded",
            "regime_distance": 0.5,
            "drift_history": list(range(0, int(drift * 10))),
            "attribution": {"top_drivers": ["pressure", "temp"], "driver_scores": {"pressure": 0.85, "temp": 0.7}},
            "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 2},
        }
        decision = engine.decide(frame)
        frames.append(decision)
        stage_sequence.append(decision.degradation_stage)

    # Verify stage progression is sensible
    assert "baseline" in stage_sequence[:5], "Early frames should be baseline"
    assert any(stage in ["early_shift", "emerging_degradation"] for stage in stage_sequence[5:15])

    # Track stage transitions
    transitions = []
    for i in range(1, len(stage_sequence)):
        if stage_sequence[i] != stage_sequence[i-1]:
            transitions.append(f"{stage_sequence[i-1]}→{stage_sequence[i]}")

    # Should have meaningful transitions
    assert len(transitions) > 0, "Long degradation should have stage transitions"
    print(f"\nStage transitions over 25 frames: {transitions}")

    # Verify stage-specific recommendations change
    recommendations = [f.stage_specific_recommendation for f in frames]
    unique_recs = set(r for r in recommendations if r)
    print(f"Unique stage-specific recommendations: {unique_recs}")
    assert len(unique_recs) > 1, "Recommendations should vary across stages"


def test_transition_events_not_repeated():
    """Verify transition events are only emitted on state change."""
    engine = DecisionEngine(enable_persistence_tracking=True)

    # Frame 1: Baseline
    frame1 = {
        "asset_id": "pump_test",
        "state": "STABLE",
        "structural_drift_score": 0.1,
        "relational_instability_score": 0.05,
        "system_phase": "stable",
        "policy_watch": False,
        "shock_activity": 0.0,
        "subsystem_instability": 0.02,
        "sensor_relationships": ["temp"],
        "regime_name": "nominal",
        "regime_distance": 0.1,
        "drift_history": [0.1],
        "attribution": {"top_drivers": [], "driver_scores": {}},
        "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 1},
    }

    decision1 = engine.decide(frame1)
    assert decision1.stage_transition_event is None  # First frame, no transition

    # Frame 2: Same as Frame 1 (no state change)
    decision2 = engine.decide(frame1)
    assert decision2.stage_transition_event is None  # Same stage, no transition

    # Frame 3: Transition to early_shift
    frame3 = {
        "asset_id": "pump_test",
        "state": "WATCH",
        "structural_drift_score": 0.4,
        "relational_instability_score": 0.3,
        "system_phase": "degrading",
        "policy_watch": True,
        "shock_activity": 0.1,
        "subsystem_instability": 0.2,
        "sensor_relationships": ["temp"],
        "regime_name": "nominal",
        "regime_distance": 0.3,
        "drift_history": [0.1, 0.2, 0.4],
        "attribution": {"top_drivers": ["temp"], "driver_scores": {"temp": 0.7}},
        "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 1},
    }

    decision3 = engine.decide(frame3)
    assert decision3.stage_transition_event is not None
    assert "baseline" in decision3.stage_transition_event
    assert "early_shift" in decision3.stage_transition_event

    # Frame 4: Same as Frame 3 (no new transition)
    decision4 = engine.decide(frame3)
    assert decision4.stage_transition_event is None  # Same stage, no new transition


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
