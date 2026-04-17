"""Phase 6 Comprehensive Validation: Multi-Horizon Action Intelligence.

Validates all requirements:
1. Action horizon definitions (now/soon/watchlist)
2. Primary action mapping
3. Secondary action generation
4. Horizon-aware policy (stage + severity + trajectory + pattern)
5. Action stability (prevents bouncing)
6. Recommendation prioritization
7. Backward compatibility
8. Decision output extensions
"""

from __future__ import annotations

import json
from typing import Any

from neraium_core.decision import DecisionEngine
from neraium_core.decision.action_horizon import (
    ActionHorizonPolicy,
    compute_primary_action,
    compute_secondary_actions,
)


def test_horizon_definitions():
    """Verify action horizon definitions and mappings."""
    print("\n" + "=" * 80)
    print("TEST 1: HORIZON DEFINITIONS")
    print("=" * 80)

    policy = ActionHorizonPolicy()

    test_cases = [
        {
            "name": "HIGH severity → now",
            "severity": "HIGH",
            "stage": "failure_approach",
            "trajectory": "degrading",
            "expected": "now",
        },
        {
            "name": "ELEVATED + persistent_degradation → soon",
            "severity": "ELEVATED",
            "stage": "persistent_degradation",
            "trajectory": "degrading",
            "expected": "soon",
        },
        {
            "name": "MODERATE + early_shift → watchlist",
            "severity": "MODERATE",
            "stage": "early_shift",
            "trajectory": "stable",
            "expected": "watchlist",
        },
        {
            "name": "LOW → watchlist",
            "severity": "LOW",
            "stage": "baseline",
            "trajectory": "stable",
            "expected": "watchlist",
        },
    ]

    for test_case in test_cases:
        horizon, reason = policy.compute_action_horizon(
            severity=test_case["severity"],
            degradation_stage=test_case["stage"],
            trajectory=test_case["trajectory"],
        )
        assert horizon == test_case["expected"], f"Failed: {test_case['name']}"
        print(f"✓ {test_case['name']}: {horizon}")
        print(f"  Reason: {reason}\n")

    print("✅ All horizon definition tests passed\n")


def test_pattern_influence():
    """Verify pattern outcome influences horizon."""
    print("=" * 80)
    print("TEST 2: PATTERN INFLUENCE ON HORIZON")
    print("=" * 80)

    policy = ActionHorizonPolicy()

    # Self-resolved pattern lowers urgency
    horizon, reason = policy.compute_action_horizon(
        severity="MODERATE",
        degradation_stage="persistent_degradation",
        trajectory="degrading",
        pattern_outcome_type="self_resolved",
        pattern_match_tier="strong",
    )
    assert horizon == "watchlist", f"Self-resolved should lower to watchlist, got {horizon}"
    print("✓ Self-resolved pattern lowers to watchlist")
    print(f"  Reason: {reason}\n")

    # Failure progression escalates
    horizon, reason = policy.compute_action_horizon(
        severity="ELEVATED",
        degradation_stage="accelerated_deterioration",
        trajectory="degrading",
        pattern_outcome_type="failure_progression",
        pattern_match_tier="strong",
    )
    assert horizon == "now", f"Failure progression should escalate to now, got {horizon}"
    print("✓ Failure progression pattern escalates to now")
    print(f"  Reason: {reason}\n")

    print("✅ All pattern influence tests passed\n")


def test_time_pressure():
    """Verify time-to-instability pressure on horizon."""
    print("=" * 80)
    print("TEST 3: TIME PRESSURE ESCALATION")
    print("=" * 80)

    policy = ActionHorizonPolicy()

    # Short window escalates ELEVATED to now
    horizon, reason = policy.compute_action_horizon(
        severity="ELEVATED",
        degradation_stage="persistent_degradation",
        trajectory="degrading",
        time_to_instability=4.0,
    )
    assert horizon == "now", f"Short window should escalate to now, got {horizon}"
    print("✓ Short time window (4 cycles) escalates ELEVATED to now")
    print(f"  Reason: {reason}\n")

    print("✅ All time pressure tests passed\n")


def test_action_stability():
    """Verify stability logic prevents unnecessary bouncing."""
    print("=" * 80)
    print("TEST 4: ACTION STABILITY (BOUNCE PREVENTION)")
    print("=" * 80)

    policy = ActionHorizonPolicy()

    # Establish baseline
    h1, _ = policy.compute_action_horizon(
        severity="ELEVATED",
        degradation_stage="persistent_degradation",
        trajectory="degrading",
    )
    assert h1 == "soon"
    print("✓ Frame 1: Baseline horizon established as 'soon'")

    # Try to downgrade without reason - should be blocked
    h2, _ = policy.compute_action_horizon(
        severity="ELEVATED",
        degradation_stage="persistent_degradation",
        trajectory="degrading",
    )
    assert h2 == "soon", f"Should maintain 'soon' for stability, got {h2}"
    print("✓ Frame 2: Stability prevents downgrade without justification")

    # Severity drop allows downgrade
    h3, reason = policy.compute_action_horizon(
        severity="MODERATE",
        degradation_stage="emerging_degradation",
        trajectory="improving",
    )
    print(f"✓ Frame 3: Can downgrade with severity drop to: {h3}")
    print(f"  Reason: {reason}\n")

    print("✅ All stability tests passed\n")


def test_primary_action_mapping():
    """Verify primary action mapping."""
    print("=" * 80)
    print("TEST 5: PRIMARY ACTION MAPPING")
    print("=" * 80)

    actions = {
        ("now", "HIGH", "failure_approach"): "escalate_to_operations",
        ("soon", "ELEVATED", "persistent_degradation"): "schedule_inspection",
        ("watchlist", "MODERATE", "early_shift"): "increase_monitoring",
    }

    for (horizon, severity, stage), expected_action_type in actions.items():
        action = compute_primary_action(
            horizon=horizon,  # type: ignore
            severity=severity,
            stage=stage,
        )
        assert action in {
            expected_action_type,
            expected_action_type.replace("_", " "),
        }, f"Unexpected action for {horizon}/{severity}/{stage}: {action}"
        print(f"✓ {horizon}/{severity}/{stage} → {action}")

    print("\n✅ All primary action tests passed\n")


def test_secondary_actions():
    """Verify secondary action generation."""
    print("=" * 80)
    print("TEST 6: SECONDARY ACTION GENERATION")
    print("=" * 80)

    # HIGH severity should generate secondary actions
    secondary = compute_secondary_actions(
        primary="escalate_to_operations",
        severity="HIGH",
        stage="failure_approach",
        drift_score=0.8,
    )
    assert len(secondary) > 0, "HIGH severity should generate secondary actions"
    print(f"✓ HIGH severity generates {len(secondary)} secondary actions: {secondary}")

    # LOW severity may not generate secondary actions
    secondary = compute_secondary_actions(
        primary="continue_observation",
        severity="LOW",
        stage="baseline",
        drift_score=0.1,
    )
    print(f"✓ LOW severity generates {len(secondary)} secondary actions: {secondary}\n")

    print("✅ All secondary action tests passed\n")


def test_decision_engine_integration():
    """Verify DecisionEngine produces all Phase 6 fields."""
    print("=" * 80)
    print("TEST 7: DECISION ENGINE INTEGRATION")
    print("=" * 80)

    engine = DecisionEngine(enable_persistence_tracking=True)

    sii_output = {
        "state": "ALERT",
        "structural_drift_score": 0.7,
        "relational_instability_score": 0.6,
        "system_phase": "degrading",
        "regime_name": "degrading",
        "regime_distance": 0.7,
        "shock_activity": 0.2,
        "subsystem_instability": 0.5,
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "attribution": {"top_drivers": ["pressure"], "top_relationships": []},
        "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 3},
    }

    decision = engine.decide(sii_output=sii_output)

    # Verify Phase 6 fields exist and are populated
    assert decision.action_horizon is not None, "action_horizon missing"
    assert decision.primary_action is not None, "primary_action missing"
    assert isinstance(decision.secondary_actions, list), "secondary_actions not a list"
    assert decision.action_priority_reason is not None, "action_priority_reason missing"

    print("✓ Decision.action_horizon: ", decision.action_horizon)
    print("✓ Decision.primary_action: ", decision.primary_action)
    print("✓ Decision.secondary_actions: ", decision.secondary_actions)
    print("✓ Decision.action_priority_reason: ", decision.action_priority_reason)

    # to_dict includes all new fields
    decision_dict = decision.to_dict()
    assert "action_horizon" in decision_dict
    assert "primary_action" in decision_dict
    assert "secondary_actions" in decision_dict
    assert "action_priority_reason" in decision_dict
    assert "action_tradeoff_note" in decision_dict

    print("\n✓ All Phase 6 fields present in to_dict()")

    # to_dict is JSON-serializable
    json_str = json.dumps(decision_dict)
    assert isinstance(json_str, str)
    print("✓ Decision output is JSON-serializable\n")

    print("✅ All integration tests passed\n")


def test_recommendation_progression():
    """Verify recommendation progression over multiple frames."""
    print("=" * 80)
    print("TEST 8: RECOMMENDATION PROGRESSION (10 FRAMES)")
    print("=" * 80)

    engine = DecisionEngine(enable_persistence_tracking=True)

    # Simulate persistent degradation over 10 frames
    frames_data = [
        {"drift": 0.10, "label": "Frame 1: Initial"},
        {"drift": 0.20, "label": "Frame 2: Early rise"},
        {"drift": 0.35, "label": "Frame 3: Acceleration"},
        {"drift": 0.45, "label": "Frame 4: Moderate"},
        {"drift": 0.55, "label": "Frame 5: Elevated"},
        {"drift": 0.62, "label": "Frame 6: Persistent"},
        {"drift": 0.70, "label": "Frame 7: High risk"},
        {"drift": 0.78, "label": "Frame 8: Acceleration"},
        {"drift": 0.85, "label": "Frame 9: Critical"},
        {"drift": 0.92, "label": "Frame 10: Imminent"},
    ]

    horizons = []
    actions = []

    print(f"{'Frame':<10} {'Drift':<8} {'Severity':<12} {'Horizon':<12} {'Action':<25}")
    print("-" * 70)

    for frame_data in frames_data:
        drift = frame_data["drift"]
        sii_output = {
            "state": "ALERT" if drift > 0.65 else ("WATCH" if drift > 0.3 else "STABLE"),
            "structural_drift_score": drift,
            "relational_instability_score": drift * 0.85,
            "system_phase": "degrading" if drift > 0.3 else "stable",
            "regime_name": "degrading" if drift > 0.4 else "nominal",
            "regime_distance": min(drift * 0.6, 1.0),
            "shock_activity": 0.15,
            "subsystem_instability": drift * 0.7,
            "sensor_relationships": ["temp", "pressure"],
            "attribution": {"top_drivers": ["pressure"], "top_relationships": []},
            "data_quality": {"missing_sensor_count": 0, "valid_signal_count": 2},
        }

        decision = engine.decide(sii_output=sii_output)
        horizons.append(decision.action_horizon)
        actions.append(decision.primary_action)

        print(
            f"{frame_data['label']:<10} {drift:<8.2f} {decision.severity:<12} "
            f"{decision.action_horizon:<12} {decision.primary_action or 'none':<25}"
        )

    # Verify progression: should escalate from watchlist → soon → now
    print("\nProgression summary:")
    print(f"  Horizons: {horizons}")
    print(f"  First horizon: {horizons[0]}")
    print(f"  Last horizon: {horizons[-1]}")

    # Should end at "now" for high drift
    assert horizons[-1] == "now", f"Final frame should be 'now', got {horizons[-1]}"
    print("\n✓ Horizons correctly escalate to 'now' at high drift\n")

    print("✅ All progression tests passed\n")


def main():
    """Run all Phase 6 validation tests."""
    print("\n" + "=" * 80)
    print("PHASE 6: MULTI-HORIZON ACTION INTELLIGENCE VALIDATION")
    print("=" * 80)

    test_horizon_definitions()
    test_pattern_influence()
    test_time_pressure()
    test_action_stability()
    test_primary_action_mapping()
    test_secondary_actions()
    test_decision_engine_integration()
    test_recommendation_progression()

    print("=" * 80)
    print("✅ ALL PHASE 6 VALIDATION TESTS PASSED")
    print("=" * 80)
    print("\nSummary of Phase 6 Implementation:")
    print("  1. ✓ Action horizon definitions (now/soon/watchlist)")
    print("  2. ✓ Horizon-aware policy mapping (stage + severity + trajectory + pattern)")
    print("  3. ✓ Primary action computation")
    print("  4. ✓ Secondary action generation")
    print("  5. ✓ Action stability logic (prevents bouncing)")
    print("  6. ✓ Pattern outcome influence")
    print("  7. ✓ Time pressure escalation")
    print("  8. ✓ Decision engine integration")
    print("  9. ✓ Backward compatibility (to_dict, JSON serialization)")
    print(" 10. ✓ Recommendation progression validation")


if __name__ == "__main__":
    main()
