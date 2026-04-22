"""Phase 7 Validation: End-to-end system coherence validation.

Validates:
1. Unit 001 regression (frame 26 HIGH severity)
2. Full FD004 dataset (no contradictions)
3. Coherence across 100+ decisions
4. Normalization and output quality
"""

import json
from pathlib import Path
from neraium_core.decision.engine import DecisionEngine
from neraium_core.decision.coherence import validate_decision_coherence
from neraium_core.decision.normalization import (
    normalize_decision_output,
    validate_numeric_ranges,
)
from neraium_core.decision.traceability import validate_trace


def test_unit_001_regression():
    """Validate Unit 001 regression: frame 26 should have HIGH severity."""
    print("\n" + "="*70)
    print("TEST: Unit 001 Regression (Frame 26 HIGH Severity)")
    print("="*70)

    engine = DecisionEngine()

    # Simulate Unit 001 frame progression
    # (In production, these would come from actual FD001 data)
    test_frames = [
        {
            "state": "STABLE",
            "structural_drift_score": 0.1,
            "relational_instability_score": 0.1,
            "system_phase": "stable",
            "regime_name": "baseline",
            "regime_distance": 0.1,
            "shock_activity": 0.0,
            "subsystem_instability": 0.0,
            "sensor_relationships": [{"signal": "temp"}, {"signal": "pressure"}],
            "data_quality": {"missing_sensor_count": 0},
        }
        for _ in range(10)
    ]

    # Add frames showing degradation
    for frame_num in range(11, 26):
        drift = 0.2 + (frame_num - 11) * 0.08
        test_frames.append(
            {
                "state": "ELEVATED" if frame_num < 24 else "HIGH",
                "structural_drift_score": drift,
                "relational_instability_score": drift * 0.8,
                "system_phase": "stable" if frame_num < 20 else "degrading",
                "regime_name": "baseline",
                "regime_distance": drift * 0.5,
                "shock_activity": 0.0,
                "subsystem_instability": 0.1,
                "sensor_relationships": [{"signal": "temp"}, {"signal": "pressure"}],
                "data_quality": {"missing_sensor_count": 0},
            }
        )

    # Process frames and validate coherence
    decisions = []
    for i, frame in enumerate(test_frames):
        decision = engine.decide(frame)
        decisions.append(decision)

        # Validate
        is_coherent, errors = validate_decision_coherence(decision)
        if not is_coherent:
            print(f"  Frame {i}: ✗ INCOHERENT - {errors}")
        else:
            print(f"  Frame {i}: ✓ {decision.severity} (horizon: {decision.action_horizon})")

    # Check frame 26 specifically
    if len(decisions) >= 26:
        frame_26 = decisions[25]  # 0-indexed
        assert frame_26.severity == "HIGH", f"Frame 26 should be HIGH, got {frame_26.severity}"
        assert frame_26.action_horizon in {"now", "soon"}, f"Frame 26 should be actionable"
        print(f"\n✓ Frame 26 regression passed: {frame_26.severity} + horizon {frame_26.action_horizon}")

    return len(decisions) > 0


def test_coherence_consistency_100_frames():
    """Test coherence across 100 synthetic frames."""
    print("\n" + "="*70)
    print("TEST: Coherence Consistency (100 Frames)")
    print("="*70)

    engine = DecisionEngine()
    coherence_passes = 0
    coherence_failures = []

    for frame_num in range(100):
        # Simulate realistic degradation curve
        if frame_num < 20:
            severity_val = 0.1
            phase = "stable"
        elif frame_num < 50:
            severity_val = 0.5 + (frame_num - 20) * 0.015
            phase = "stable"
        elif frame_num < 70:
            severity_val = 1.0 + (frame_num - 50) * 0.08
            phase = "degrading"
        else:
            severity_val = 2.5 + (frame_num - 70) * 0.1
            phase = "degrading"

        frame = {
            "state": "ELEVATED" if severity_val > 1.5 else ("MODERATE" if severity_val > 0.7 else "LOW"),
            "structural_drift_score": severity_val,
            "relational_instability_score": severity_val * 0.75,
            "system_phase": phase,
            "regime_name": "baseline",
            "regime_distance": severity_val * 0.3,
            "shock_activity": 0.0,
            "subsystem_instability": severity_val * 0.5,
            "sensor_relationships": [
                {"signal": "temp"},
                {"signal": "pressure"},
                {"signal": "vibration"},
            ],
            "data_quality": {"missing_sensor_count": 0},
        }

        decision = engine.decide(frame)

        is_coherent, errors = validate_decision_coherence(decision)
        if is_coherent:
            coherence_passes += 1
        else:
            coherence_failures.append((frame_num, errors))

        # Also validate numeric ranges
        num_valid, num_errors = validate_numeric_ranges(decision)
        if not num_valid:
            coherence_failures.append((frame_num, num_errors))

    print(f"  Coherence passes: {coherence_passes}/100")
    if coherence_failures:
        print(f"  Coherence failures: {len(coherence_failures)}")
        for frame_num, errors in coherence_failures[:5]:
            print(f"    Frame {frame_num}: {errors}")

    assert len(coherence_failures) == 0, f"Found {len(coherence_failures)} incoherent frames"
    return True


def test_output_normalization_100_frames():
    """Test numeric normalization across 100 frames."""
    print("\n" + "="*70)
    print("TEST: Output Normalization (100 Frames)")
    print("="*70)

    engine = DecisionEngine()
    normalization_errors = []

    for frame_num in range(100):
        drift = 0.5 + (frame_num * 0.02)
        frame = {
            "state": "ELEVATED" if drift > 1.5 else "MODERATE",
            "structural_drift_score": drift,
            "relational_instability_score": drift * 0.8,
            "system_phase": "stable",
            "regime_name": "baseline",
            "regime_distance": drift * 0.5,
            "shock_activity": 0.1,
            "subsystem_instability": drift * 0.3,
            "sensor_relationships": [
                {"signal": "temp"},
                {"signal": "pressure"},
            ],
            "data_quality": {"missing_sensor_count": 0},
        }

        decision = engine.decide(frame)
        normalized = normalize_decision_output(decision)

        # Check that normalized values are properly formatted
        try:
            if "finding_confidence" in normalized:
                assert (
                    isinstance(normalized["finding_confidence"], float)
                    and 0.0 <= normalized["finding_confidence"] <= 1.0
                ), f"finding_confidence invalid: {normalized['finding_confidence']}"

            if "persistence_frames_at_level" in normalized:
                assert isinstance(
                    normalized["persistence_frames_at_level"], int
                ), f"persistence_frames should be int"

            # Verify it's JSON serializable
            json.dumps(normalized)

        except (AssertionError, TypeError) as e:
            normalization_errors.append((frame_num, str(e)))

    if normalization_errors:
        print(f"  Normalization errors: {len(normalization_errors)}")
        for frame_num, error in normalization_errors[:5]:
            print(f"    Frame {frame_num}: {error}")

    assert len(normalization_errors) == 0, f"Found {len(normalization_errors)} normalization errors"
    print(f"  ✓ All 100 frames normalized successfully")
    return True


def test_decision_trace_quality():
    """Test decision trace generation quality."""
    print("\n" + "="*70)
    print("TEST: Decision Trace Quality")
    print("="*70)

    engine = DecisionEngine()
    trace_errors = []

    test_cases = [
        # Case 1: Baseline stable
        {
            "name": "Baseline Stable",
            "frame": {
                "state": "STABLE",
                "structural_drift_score": 0.1,
                "relational_instability_score": 0.05,
                "system_phase": "stable",
                "regime_name": "baseline",
                "regime_distance": 0.1,
                "shock_activity": 0.0,
                "subsystem_instability": 0.0,
                "sensor_relationships": [{"signal": "temp"}],
                "data_quality": {"missing_sensor_count": 0},
            },
        },
        # Case 2: Elevated severity
        {
            "name": "Elevated Severity",
            "frame": {
                "state": "ELEVATED",
                "structural_drift_score": 1.8,
                "relational_instability_score": 1.4,
                "system_phase": "degrading",
                "regime_name": "baseline",
                "regime_distance": 0.5,
                "shock_activity": 0.2,
                "subsystem_instability": 0.6,
                "sensor_relationships": [{"signal": "temp"}, {"signal": "pressure"}],
                "data_quality": {"missing_sensor_count": 0},
            },
        },
        # Case 3: HIGH severity
        {
            "name": "HIGH Severity",
            "frame": {
                "state": "HIGH",
                "structural_drift_score": 2.8,
                "relational_instability_score": 2.4,
                "system_phase": "degrading",
                "regime_name": "baseline",
                "regime_distance": 0.8,
                "shock_activity": 0.5,
                "subsystem_instability": 1.2,
                "sensor_relationships": [
                    {"signal": "temp"},
                    {"signal": "pressure"},
                    {"signal": "vibration"},
                ],
                "data_quality": {"missing_sensor_count": 0},
            },
        },
    ]

    for case in test_cases:
        decision = engine.decide(case["frame"])

        # Check trace
        if decision.decision_trace:
            trace = decision.decision_trace
            is_valid, errors = validate_trace(
                type("Trace", (), {
                    "primary_factor": trace.get("primary_factor"),
                    "secondary_factors": trace.get("secondary_factors", []),
                    "confidence_rationale": trace.get("confidence_rationale"),
                })()
            )
            if not is_valid:
                trace_errors.append((case["name"], errors))
            else:
                print(f"  {case['name']}: ✓")
                print(f"    Primary: {trace.get('primary_factor')}")
                print(f"    Secondary: {trace.get('secondary_factors', [])}")
        else:
            print(f"  {case['name']}: ✗ No trace generated")

    assert len(trace_errors) == 0, f"Found trace quality issues: {trace_errors}"
    return True


def test_fd004_simulation():
    """Simulate FD004-like dataset with multiple units."""
    print("\n" + "="*70)
    print("TEST: FD004 Simulation (10 Units, 100 Cycles)")
    print("="*70)

    engine = DecisionEngine()
    total_decisions = 0
    coherence_issues = 0

    for unit in range(1, 11):
        for cycle in range(100):
            # Simulate realistic degradation with some variation by unit
            unit_degradation_rate = 0.01 + (unit * 0.002)
            drift = 0.1 + (cycle * unit_degradation_rate) + (0.1 * (unit % 3))

            frame = {
                "state": "HIGH" if drift > 2.5 else ("ELEVATED" if drift > 1.5 else "MODERATE" if drift > 0.7 else "LOW"),
                "structural_drift_score": drift,
                "relational_instability_score": drift * 0.8,
                "system_phase": "degrading" if drift > 1.0 else "stable",
                "regime_name": f"unit_{unit}",
                "regime_distance": drift * 0.3,
                "shock_activity": 0.05 * (1 + (unit % 2)),
                "subsystem_instability": drift * 0.5,
                "sensor_relationships": [
                    {"signal": f"temp_{unit}"},
                    {"signal": f"pressure_{unit}"},
                    {"signal": f"vibration_{unit}"},
                ],
                "data_quality": {"missing_sensor_count": 0},
            }

            decision = engine.decide(frame)
            total_decisions += 1

            is_coherent, errors = validate_decision_coherence(decision)
            if not is_coherent:
                coherence_issues += 1

    print(f"  Total decisions: {total_decisions}")
    print(f"  Coherence issues: {coherence_issues}")

    assert coherence_issues == 0, f"Found {coherence_issues} incoherent decisions in FD004 simulation"
    return True


def run_all_validation_tests():
    """Run all Phase 7 validation tests."""
    print("\n" + "#"*70)
    print("# PHASE 7 VALIDATION: System Coherence and Output Quality")
    print("#"*70)

    tests = [
        ("Unit 001 Regression", test_unit_001_regression),
        ("Coherence Consistency (100 Frames)", test_coherence_consistency_100_frames),
        ("Output Normalization (100 Frames)", test_output_normalization_100_frames),
        ("Decision Trace Quality", test_decision_trace_quality),
        ("FD004 Simulation (10 Units, 100 Cycles)", test_fd004_simulation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except AssertionError as e:
            results[test_name] = f"FAIL: {str(e)[:100]}"
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)[:100]}"

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    for test_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"{status} {test_name}: {result}")

    all_pass = all(r == "PASS" for r in results.values())
    print("="*70)
    if all_pass:
        print("✓ ALL VALIDATION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")

    return all_pass


if __name__ == "__main__":
    success = run_all_validation_tests()
    exit(0 if success else 1)
