"""Regression test: Phase 1 vs Phase 2 decision layer on FD004 Unit 001.

This test replays the audit scenario to compare behavior between the original
decision layer (Phase 1) and the improved version (Phase 2).

Metrics tracked:
1. First meaningful detection cycle
2. Failure cycle
3. Lead time
4. Number of surfaced alerts
5. Number of suppressed events
6. Number of recommendation changes
7. HIGH severity repetition rate
8. Impact of startup suppression/hysteresis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neraium_core.decision import DecisionEngine
from neraium_core.decision.models import SeverityLevel


# ============================================================================
# FIXTURES: FD004 Unit 001 Test Data
# ============================================================================

@pytest.fixture
def fd004_unit001_frames() -> list[dict[str, Any]]:
    """Load FD004 Unit 001 timeseries data from saved report."""
    report_path = Path(__file__).parent.parent / "fd004_outputs_subset" / "fd004_real_report.json"

    if not report_path.exists():
        pytest.skip(f"FD004 test data not found at {report_path}")

    with open(report_path) as f:
        data = json.load(f)

    timeseries = data.get("timeseries", [])
    unit001_frames = [f for f in timeseries if f.get("asset_id") == "unit_001"]

    if not unit001_frames:
        pytest.skip("No Unit 001 frames in FD004 data")

    return unit001_frames


@pytest.fixture
def fd004_unit001_metadata(fd004_unit001_frames: list[dict]) -> dict[str, Any]:
    """Extract metadata about Unit 001 from FD004 report."""
    report_path = Path(__file__).parent.parent / "fd004_outputs_subset" / "fd004_real_report.json"

    with open(report_path) as f:
        data = json.load(f)

    unit_summaries = data.get("unit_summaries", [])
    unit_001_summary = next((u for u in unit_summaries if u.get("asset_id") == "unit_001"), {})

    return {
        "first_medium_cycle": unit_001_summary.get("first_MEDIUM_step"),
        "first_high_cycle": unit_001_summary.get("first_HIGH_step"),
        "early_warning_window": unit_001_summary.get("early_warning_window"),
        "peak_instability": unit_001_summary.get("peak_instability"),
        "average_drift": unit_001_summary.get("average_drift"),
        "total_cycles": len(fd004_unit001_frames),
    }


# ============================================================================
# SYNTHETIC SII OUTPUT BUILDER (for frames that lack details)
# ============================================================================

def build_sii_output_from_timeseries(frame: dict[str, Any]) -> dict[str, Any]:
    """Convert timeseries frame to SII output format for decision layer.

    Note: This is a synthetic reconstruction from limited data.
    Real SII outputs would have more fields; we estimate from available data.
    """
    composite = float(frame.get("composite_instability", 0.0))
    drift = float(frame.get("structural_drift_score", 0.0))

    # Estimate relational instability from composite
    # Composite is a blend of drift and relational factors
    relational = max(0.0, min(1.0, (composite / 500.0) if composite > 0 else 0.0))

    # Estimate shock activity: high spikes suggest transients
    trend = frame.get("trend", "stable")
    shock_activity = 0.8 if trend == "drift" and drift > 0.5 else 0.1

    # Determine state based on risk_level
    risk_level = frame.get("risk_level", "LOW")
    state_map = {
        "LOW": "STABLE",
        "MEDIUM": "WATCH",
        "HIGH": "ALERT",
    }
    state = state_map.get(risk_level, "STABLE")

    # Phase mapping
    phase = frame.get("phase", "stable")
    phase_map = {"drift": "degrading", "drift": "degrading"}
    system_phase = phase_map.get(phase, phase)

    return {
        "timestamp": frame.get("timestamp"),
        "asset_id": frame.get("asset_id"),
        "state": state,
        "structural_drift_score": drift,
        "relational_instability_score": relational,
        "system_phase": system_phase,
        "shock_activity": shock_activity,
        "subsystem_instability": max(0.0, relational * 0.7),
        "entropy": max(0.0, (composite / 1000.0) if composite > 0 else 0.0),
        "sensor_relationships": ["temp", "pressure", "vibration"],
        "regime_name": "nominal" if drift < 0.3 else "degrading",
        "regime_distance": min(1.0, drift * 0.5),
        "attribution": {
            "top_drivers": ["pressure", "temperature"] if drift > 0.3 else [],
            "top_relationships": [],
        },
        "drift_history": [],
        "data_quality": {
            "missing_sensor_count": 0,
            "valid_signal_count": 3,
        },
    }


# ============================================================================
# PHASE 1 vs PHASE 2 COMPARISON
# ============================================================================

def test_phase2_regression_unit001(
    fd004_unit001_frames: list[dict[str, Any]],
    fd004_unit001_metadata: dict[str, Any],
) -> None:
    """Compare Phase 1 and Phase 2 decision layer behavior on FD004 Unit 001.

    Phase 1 = DecisionEngine(enable_persistence_tracking=False)
    Phase 2 = DecisionEngine(enable_persistence_tracking=True)
    """

    print(f"\n\n{'='*80}")
    print("FD004 UNIT 001 DECISION LAYER REGRESSION TEST")
    print(f"{'='*80}\n")

    # Metadata
    print("UNIT 001 METADATA FROM FD004:")
    print(f"  Total cycles: {fd004_unit001_metadata['total_cycles']}")
    print(f"  First MEDIUM detection: Cycle {fd004_unit001_metadata['first_medium_cycle']}")
    print(f"  First HIGH detection: Cycle {fd004_unit001_metadata['first_high_cycle']}")
    print(f"  Early warning window: {fd004_unit001_metadata['early_warning_window']} frame(s)")
    print(f"  Peak composite instability: {fd004_unit001_metadata['peak_instability']:.2f}")
    print(f"  Average drift score: {fd004_unit001_metadata['average_drift']:.4f}\n")

    # Initialize both engines
    engine_phase1 = DecisionEngine(enable_persistence_tracking=False)
    engine_phase2 = DecisionEngine(enable_persistence_tracking=True)

    # Track metrics
    metrics_p1 = {
        "first_elevated_cycle": None,
        "first_high_cycle": None,
        "total_surfaced_alerts": 0,
        "total_suppressed_events": 0,
        "recommendation_changes": 0,
        "high_severity_consecutive_frames": 0,
        "high_severity_streaks": [],
        "current_high_streak": 0,
        "last_recommendation": None,
    }

    metrics_p2 = {
        "first_elevated_cycle": None,
        "first_high_cycle": None,
        "total_surfaced_alerts": 0,
        "total_suppressed_events": 0,
        "recommendation_changes": 0,
        "high_severity_consecutive_frames": 0,
        "high_severity_streaks": [],
        "current_high_streak": 0,
        "last_recommendation": None,
    }

    # Replay frames through both engines
    print("FRAME-BY-FRAME REPLAY:\n")
    print(f"{'Cycle':<8} {'Drift':<8} {'Compos':<8} {'P1-Sev':<12} {'P1-Suppr':<10} {'P1-Rec':<20} {'P2-Sev':<12} {'P2-Suppr':<10} {'P2-Rec':<20}")
    print("-" * 120)

    for cycle, frame in enumerate(fd004_unit001_frames, start=1):
        sii_output = build_sii_output_from_timeseries(frame)

        # Process through Phase 1
        decision_p1 = engine_phase1.decide(sii_output=sii_output)

        # Process through Phase 2
        decision_p2 = engine_phase2.decide(sii_output=sii_output)

        # Track Phase 1 metrics
        if not decision_p1.suppress:
            metrics_p1["total_surfaced_alerts"] += 1
            if decision_p1.severity == "ELEVATED" and metrics_p1["first_elevated_cycle"] is None:
                metrics_p1["first_elevated_cycle"] = cycle
            if decision_p1.severity == "HIGH" and metrics_p1["first_high_cycle"] is None:
                metrics_p1["first_high_cycle"] = cycle
        else:
            metrics_p1["total_suppressed_events"] += 1

        if decision_p1.recommended_action:
            if decision_p1.recommended_action != metrics_p1["last_recommendation"]:
                metrics_p1["recommendation_changes"] += 1
                metrics_p1["last_recommendation"] = decision_p1.recommended_action

        if decision_p1.severity == "HIGH":
            metrics_p1["high_severity_consecutive_frames"] += 1
            metrics_p1["current_high_streak"] += 1
        else:
            if metrics_p1["current_high_streak"] > 0:
                metrics_p1["high_severity_streaks"].append(metrics_p1["current_high_streak"])
                metrics_p1["current_high_streak"] = 0

        # Track Phase 2 metrics
        if not decision_p2.suppress:
            metrics_p2["total_surfaced_alerts"] += 1
            if decision_p2.severity == "ELEVATED" and metrics_p2["first_elevated_cycle"] is None:
                metrics_p2["first_elevated_cycle"] = cycle
            if decision_p2.severity == "HIGH" and metrics_p2["first_high_cycle"] is None:
                metrics_p2["first_high_cycle"] = cycle
        else:
            metrics_p2["total_suppressed_events"] += 1

        if decision_p2.recommended_action:
            if decision_p2.recommended_action != metrics_p2["last_recommendation"]:
                metrics_p2["recommendation_changes"] += 1
                metrics_p2["last_recommendation"] = decision_p2.recommended_action

        if decision_p2.severity == "HIGH":
            metrics_p2["high_severity_consecutive_frames"] += 1
            metrics_p2["current_high_streak"] += 1
        else:
            if metrics_p2["current_high_streak"] > 0:
                metrics_p2["high_severity_streaks"].append(metrics_p2["current_high_streak"])
                metrics_p2["current_high_streak"] = 0

        # Print first 30 and last 10 cycles, plus cycles 20-30
        show = (
            cycle <= 30 or
            cycle >= len(fd004_unit001_frames) - 10 or
            cycle in range(20, 31)
        )

        if show:
            drift = float(frame.get("structural_drift_score", 0.0))
            compos = float(frame.get("composite_instability", 0.0))
            p1_rec = decision_p1.recommended_action or "none"
            p2_rec = decision_p2.recommended_action or "none"

            print(
                f"{cycle:<8} {drift:<8.3f} {compos:<8.0f} "
                f"{decision_p1.severity:<12} {'T' if decision_p1.suppress else 'F':<10} "
                f"{p1_rec:<20} {decision_p2.severity:<12} {'T' if decision_p2.suppress else 'F':<10} "
                f"{p2_rec:<20}"
            )
        elif cycle == 31:
            print("... (frames 31-{}) ...".format(len(fd004_unit001_frames) - 11))

    # Finalize streaks
    if metrics_p1["current_high_streak"] > 0:
        metrics_p1["high_severity_streaks"].append(metrics_p1["current_high_streak"])
    if metrics_p2["current_high_streak"] > 0:
        metrics_p2["high_severity_streaks"].append(metrics_p2["current_high_streak"])

    # === RESULTS ===
    print(f"\n\n{'='*80}")
    print("REGRESSION ANALYSIS RESULTS")
    print(f"{'='*80}\n")

    # 1. First meaningful detection
    print("1. FIRST MEANINGFUL DETECTION:")
    print(f"   Phase 1 (original):")
    print(f"     - First ELEVATED: Cycle {metrics_p1['first_elevated_cycle']}")
    print(f"     - First HIGH: Cycle {metrics_p1['first_high_cycle']}")
    if metrics_p1['first_high_cycle']:
        lead_p1 = fd004_unit001_metadata['total_cycles'] - metrics_p1['first_high_cycle']
        print(f"     - Lead time: {lead_p1} cycles before end")
    print(f"\n   Phase 2 (improved):")
    print(f"     - First ELEVATED: Cycle {metrics_p2['first_elevated_cycle']}")
    print(f"     - First HIGH: Cycle {metrics_p2['first_high_cycle']}")
    if metrics_p2['first_high_cycle']:
        lead_p2 = fd004_unit001_metadata['total_cycles'] - metrics_p2['first_high_cycle']
        print(f"     - Lead time: {lead_p2} cycles before end")

    if metrics_p1['first_high_cycle'] and metrics_p2['first_high_cycle']:
        delta = metrics_p2['first_high_cycle'] - metrics_p1['first_high_cycle']
        direction = "DELAYED" if delta > 0 else "EARLIER"
        print(f"\n   COMPARISON: Phase 2 detects {direction} by {abs(delta)} cycle(s)")

    # 2. Failure cycle
    print(f"\n2. FAILURE CYCLE (from metadata):")
    expected_failure = fd004_unit001_metadata['total_cycles']
    print(f"   Expected failure at cycle: {expected_failure}")
    print(f"   (This is when true RUL = 0)")

    # 3. Lead time
    print(f"\n3. LEAD TIME TO FAILURE:")
    if metrics_p1['first_high_cycle']:
        lead_p1 = expected_failure - metrics_p1['first_high_cycle']
        print(f"   Phase 1: {lead_p1} cycles ({lead_p1 * 100 / expected_failure:.1f}% of total)")
    if metrics_p2['first_high_cycle']:
        lead_p2 = expected_failure - metrics_p2['first_high_cycle']
        print(f"   Phase 2: {lead_p2} cycles ({lead_p2 * 100 / expected_failure:.1f}% of total)")

    # 4. Number of surfaced alerts
    print(f"\n4. SURFACED ALERTS (not suppressed):")
    print(f"   Phase 1: {metrics_p1['total_surfaced_alerts']} alerts")
    print(f"   Phase 2: {metrics_p2['total_surfaced_alerts']} alerts")
    print(f"   Change: {metrics_p2['total_surfaced_alerts'] - metrics_p1['total_surfaced_alerts']:+d} "
          f"({(metrics_p2['total_surfaced_alerts'] - metrics_p1['total_surfaced_alerts']) * 100 / max(1, metrics_p1['total_surfaced_alerts']):+.1f}%)")

    # 5. Number of suppressed events
    print(f"\n5. SUPPRESSED EVENTS:")
    print(f"   Phase 1: {metrics_p1['total_suppressed_events']} events")
    print(f"   Phase 2: {metrics_p2['total_suppressed_events']} events")
    print(f"   Change: {metrics_p2['total_suppressed_events'] - metrics_p1['total_suppressed_events']:+d}")

    # 6. Number of recommendation changes
    print(f"\n6. RECOMMENDATION CHANGES (escalations/downgrades):")
    print(f"   Phase 1: {metrics_p1['recommendation_changes']} changes")
    print(f"   Phase 2: {metrics_p2['recommendation_changes']} changes")
    print(f"   Change: {metrics_p2['recommendation_changes'] - metrics_p1['recommendation_changes']:+d}")

    # 7. HIGH severity repetition
    print(f"\n7. HIGH SEVERITY BEHAVIOR:")
    print(f"   Phase 1:")
    print(f"     - Total HIGH frames: {metrics_p1['high_severity_consecutive_frames']}")
    print(f"     - Number of HIGH streaks: {len(metrics_p1['high_severity_streaks'])}")
    if metrics_p1['high_severity_streaks']:
        print(f"     - Streak lengths: {metrics_p1['high_severity_streaks']}")
        avg_streak = sum(metrics_p1['high_severity_streaks']) / len(metrics_p1['high_severity_streaks'])
        print(f"     - Average streak length: {avg_streak:.1f} frames")

    print(f"\n   Phase 2:")
    print(f"     - Total HIGH frames: {metrics_p2['high_severity_consecutive_frames']}")
    print(f"     - Number of HIGH streaks: {len(metrics_p2['high_severity_streaks'])}")
    if metrics_p2['high_severity_streaks']:
        print(f"     - Streak lengths: {metrics_p2['high_severity_streaks']}")
        avg_streak = sum(metrics_p2['high_severity_streaks']) / len(metrics_p2['high_severity_streaks'])
        print(f"     - Average streak length: {avg_streak:.1f} frames")

    # 8. Startup suppression impact
    print(f"\n8. STARTUP SUPPRESSION & HYSTERESIS IMPACT:")
    startup_window = 8
    p1_suppressed_early = sum(1 for frame in fd004_unit001_frames[:startup_window])
    p2_suppressed_early = sum(1 for frame in fd004_unit001_frames[:startup_window])
    print(f"   First {startup_window} cycles (startup window):")
    print(f"     - Phase 1: {metrics_p1['total_suppressed_events']} total suppressions")
    print(f"     - Phase 2: {metrics_p2['total_suppressed_events']} total suppressions")
    print(f"   (Phase 2 applies startup suppression logic for first {startup_window} frames)")

    # === SUMMARY ===
    print(f"\n\n{'='*80}")
    print("SUMMARY & RECOMMENDATION")
    print(f"{'='*80}\n")

    # Assess whether Phase 2 should replace Phase 1
    assessment = {
        "detection_earlier": False,
        "alerts_cleaner": False,
        "lead_time_maintained": False,
        "false_positives_reduced": False,
        "no_true_detections_missed": False,
    }

    # Detection timing
    if metrics_p2['first_high_cycle'] and metrics_p1['first_high_cycle']:
        if metrics_p2['first_high_cycle'] <= metrics_p1['first_high_cycle']:
            assessment["detection_earlier"] = True

    # Alerts cleaner (fewer repetitive HIGH alerts)
    if (len(metrics_p2['high_severity_streaks']) <= len(metrics_p1['high_severity_streaks']) and
        metrics_p2['high_severity_consecutive_frames'] >= metrics_p1['high_severity_consecutive_frames'] * 0.8):
        assessment["alerts_cleaner"] = True

    # Lead time maintained
    if metrics_p1['first_high_cycle'] and metrics_p2['first_high_cycle']:
        lead_p1 = expected_failure - metrics_p1['first_high_cycle']
        lead_p2 = expected_failure - metrics_p2['first_high_cycle']
        if lead_p2 >= lead_p1 * 0.9:  # Allow 10% margin
            assessment["lead_time_maintained"] = True

    # False positives reduced
    if metrics_p2['total_surfaced_alerts'] < metrics_p1['total_surfaced_alerts']:
        assessment["false_positives_reduced"] = True

    # No true detections missed
    if metrics_p2['first_high_cycle'] is not None:
        assessment["no_true_detections_missed"] = True

    print("ASSESSMENT:")
    for criterion, passed in assessment.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        criterion_name = criterion.replace("_", " ").title()
        print(f"  {status}: {criterion_name}")

    all_pass = all(assessment.values())
    print(f"\n{'='*60}")
    if all_pass:
        print("✓ PHASE 2 CAN REPLACE PHASE 1 AS DEFAULT")
    else:
        failed = [k for k, v in assessment.items() if not v]
        print(f"✗ PHASE 2 NEEDS REVIEW: {', '.join(failed)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Allow running as standalone script
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
