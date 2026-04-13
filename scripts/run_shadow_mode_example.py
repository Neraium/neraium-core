#!/usr/bin/env python3
"""Example: Running shadow mode on sample telemetry and comparing results.

This script demonstrates:
1. Processing telemetry through two engine versions in shadow mode
2. Generating summary reports
3. Comparing outputs via replay diff
4. Exporting evidence to CSV
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.shadow_mode.runner import ShadowModeRunner
from validation.shadow_mode.report import generate_report_from_evidence
from validation.shadow_mode.replay_diff import generate_replay_diff
from validation.shadow_mode.evidence import EvidenceDataFrame


def create_sample_telemetry(count: int = 100) -> list[Dict[str, Any]]:
    """Generate sample telemetry data."""
    frames = []
    for i in range(count):
        asset_id = f"turbine_{(i % 5) + 1:03d}"
        # Simulate drift signal over time
        base_drift = 0.1 + (i / count) * 0.3
        unit_drift = (i % 10) * 0.05
        drift = base_drift + unit_drift

        frames.append(
            {
                "asset_id": asset_id,
                "unit_id": f"unit_{(i % 3) + 1}",
                "timestamp": f"2024-04-12T{(i // 60):02d}:{(i % 60):02d}:00Z",
                "frame_index": i,
                "domain": "wind",
                "system_type": "generator",
                "drift_signal": drift,
                "vibration_x": 0.5 + drift,
                "vibration_y": 0.3 + drift * 0.5,
                "temperature": 60.0 + drift * 20,
            }
        )
    return frames


def decision_fn_v1(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Engine v1: Conservative thresholds."""
    drift = frame.get("drift_signal", 0.1)
    state = "ALERT" if drift > 0.35 else ("WATCH" if drift > 0.20 else "STABLE")

    return {
        "state": state,
        "policy_state": state,
        "structural_drift_score": drift,
        "structural_drift_score_smoothed": drift * 0.95,
        "relational_instability_score": drift * 0.4,
        "transition_pressure": 0.0,
        "transition_detected": False,
        "transition_state": "NONE",
        "system_health": max(0, 100 - drift * 100),
        "active_sensor_count": 3,
        "missing_sensor_count": 0,
        "data_quality_summary": {"status": "good"},
        "dominant_driver": "vibration_x" if drift > 0.1 else "temperature",
        "attribution": {
            "top_drivers": [{"name": "vibration_x", "score": 0.7}]
        },
    }


def decision_fn_v2(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Engine v2: Optimized thresholds."""
    drift = frame.get("drift_signal", 0.1)
    # Slightly different logic
    state = (
        "ALERT"
        if drift > 0.38
        else ("WATCH" if drift > 0.22 else "STABLE")
    )

    return {
        "state": state,
        "policy_state": state,
        "structural_drift_score": drift * 1.02,  # Calibration adjustment
        "structural_drift_score_smoothed": drift * 0.92,
        "relational_instability_score": drift * 0.42,
        "transition_pressure": 0.0,
        "transition_detected": False,
        "transition_state": "NONE",
        "system_health": max(0, 100 - drift * 95),
        "active_sensor_count": 3,
        "missing_sensor_count": 0,
        "data_quality_summary": {"status": "good"},
        "dominant_driver": "vibration_x" if drift > 0.12 else "temperature",
        "attribution": {
            "top_drivers": [{"name": "vibration_x", "score": 0.72}]
        },
    }


def main():
    """Run shadow mode example."""
    print("=" * 70)
    print("Shadow Mode Example: Comparing Engine Versions")
    print("=" * 70)

    # Setup
    output_dir = Path("/tmp/shadow_mode_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry = create_sample_telemetry(100)
    print(f"\n1. Generated {len(telemetry)} sample telemetry frames")

    # Run v1
    print("\n2. Processing through Engine v1...")
    runner_v1 = ShadowModeRunner(
        decision_fn=decision_fn_v1,
        output_dir=str(output_dir),
        run_name="engine_v1",
        engine_version="1.0.0",
        doctrine_version="2024-03",
        config={"thresholds": "conservative"},
    )
    runner_v1.process_telemetry_stream(telemetry)
    result_v1 = runner_v1.finalize()
    print(f"   ✓ Processed {result_v1['summary_stats']['total_frames']} frames")
    print(f"   ✓ Evidence: {result_v1['paths']['evidence_log']}")

    # Run v2
    print("\n3. Processing through Engine v2...")
    runner_v2 = ShadowModeRunner(
        decision_fn=decision_fn_v2,
        output_dir=str(output_dir),
        run_name="engine_v2",
        engine_version="2.0.0",
        doctrine_version="2024-04",
        config={"thresholds": "optimized"},
    )
    runner_v2.process_telemetry_stream(telemetry)
    result_v2 = runner_v2.finalize()
    print(f"   ✓ Processed {result_v2['summary_stats']['total_frames']} frames")
    print(f"   ✓ Evidence: {result_v2['paths']['evidence_log']}")

    # Generate reports
    print("\n4. Generating reports...")
    report_v1 = generate_report_from_evidence(
        Path(result_v1['paths']['evidence_log']),
        output_dir / "engine_v1_report.json",
    )
    report_v2 = generate_report_from_evidence(
        Path(result_v2['paths']['evidence_log']),
        output_dir / "engine_v2_report.json",
    )
    print(f"   ✓ v1 Report: {output_dir / 'engine_v1_report.json'}")
    print(f"   ✓ v2 Report: {output_dir / 'engine_v2_report.json'}")

    # Compare
    print("\n5. Comparing Engine v1 vs v2...")
    diff_report = generate_replay_diff(
        Path(result_v1['paths']['evidence_log']),
        Path(result_v2['paths']['evidence_log']),
        "Engine v1.0",
        "Engine v2.0",
        output_dir / "replay_diff.json",
    )
    print(
        f"   ✓ Match rate: {diff_report['summary']['match_rate']}%"
    )
    print(
        f"   ✓ Frames with differences: {diff_report['summary']['frames_with_differences']}"
    )

    # Show critical differences
    if diff_report['field_level_diff']['critical_field_mismatches']:
        print("\n   Critical field mismatches:")
        for field in diff_report['field_level_diff']['critical_field_mismatches']:
            print(f"     - {field['field']}: {field['mismatch_count']} mismatches")

    # Export to CSV
    print("\n6. Exporting evidence to CSV...")
    rows_v1 = EvidenceDataFrame.export_to_csv(
        Path(result_v1['paths']['evidence_log']),
        output_dir / "engine_v1_evidence.csv",
    )
    rows_v2 = EvidenceDataFrame.export_to_csv(
        Path(result_v2['paths']['evidence_log']),
        output_dir / "engine_v2_evidence.csv",
    )
    print(f"   ✓ v1 CSV: {rows_v1} rows → {output_dir / 'engine_v1_evidence.csv'}")
    print(f"   ✓ v2 CSV: {rows_v2} rows → {output_dir / 'engine_v2_evidence.csv'}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        size = file.stat().st_size if file.is_file() else 0
        print(f"  - {file.name:<40} ({size:,} bytes)")

    print("\nKey statistics:")
    print(f"  v1 total frames:          {report_v1['overview']['total_frames']}")
    print(f"  v1 unique assets:         {report_v1['overview']['unique_assets']}")
    print(
        f"  v1 STABLE frames:         {report_v1['state_distribution']['state_counts'].get('STABLE', 0)}"
    )
    print(
        f"  v1 WATCH frames:          {report_v1['state_distribution']['state_counts'].get('WATCH', 0)}"
    )
    print(
        f"  v1 ALERT frames:          {report_v1['state_distribution']['state_counts'].get('ALERT', 0)}"
    )
    print()
    print(f"  v2 total frames:          {report_v2['overview']['total_frames']}")
    print(f"  v2 unique assets:         {report_v2['overview']['unique_assets']}")
    print(
        f"  v2 STABLE frames:         {report_v2['state_distribution']['state_counts'].get('STABLE', 0)}"
    )
    print(
        f"  v2 WATCH frames:          {report_v2['state_distribution']['state_counts'].get('WATCH', 0)}"
    )
    print(
        f"  v2 ALERT frames:          {report_v2['state_distribution']['state_counts'].get('ALERT', 0)}"
    )

    print("\nComparison:")
    print(f"  Match rate:               {diff_report['summary']['match_rate']}%")
    print(
        f"  Frames with differences:  {diff_report['summary']['frames_with_differences']}"
    )

    print("\n✅ Example completed successfully!")
    print("\nNext steps:")
    print("1. Review the summary reports:")
    print(f"   - {output_dir / 'engine_v1_report.json'}")
    print(f"   - {output_dir / 'engine_v2_report.json'}")
    print("2. Analyze the replay diff:")
    print(f"   - {output_dir / 'replay_diff.json'}")
    print("3. Export evidence to your analytics tool:")
    print(f"   - {output_dir / 'engine_v1_evidence.csv'}")
    print(f"   - {output_dir / 'engine_v2_evidence.csv'}")


if __name__ == "__main__":
    main()
