#!/usr/bin/env python3
"""
Run the Neraium intelligence layer through tests.

Usage:
  python run_intel_layer_tests.py           # run full intel-layer test suite
  python run_intel_layer_tests.py --quick   # upgrade scenarios only
  python run_intel_layer_tests.py --demo    # one-shot demo, no pytest
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Run intelligence layer tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only upgrade scenarios (test_upgrade_scenarios.py)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a short demo (no pytest), print sample output",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose pytest output",
    )
    args = parser.parse_args()

    if args.demo:
        return _run_demo()

    # Pytest-based test run
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_upgrade_scenarios.py",
        "tests/test_decision_layer.py",
        "tests/test_multinode_persistence_hysteresis.py",
        "tests/test_single_sensor_stream.py",
        "tests/test_scoring.py",
    ]
    if args.quick:
        cmd = [sys.executable, "-m", "pytest", "tests/test_upgrade_scenarios.py"]
    if args.verbose:
        cmd.append("-v")
    cmd.append("--tb=short")
    if not args.verbose:
        cmd.append("-q")

    result = subprocess.run(cmd, cwd=".")
    return result.returncode


def _run_demo() -> int:
    """One-shot demo: run a few frames through the engine and print key intel outputs."""
    import math
    import tempfile
    from neraium_core.alignment import StructuralEngine

    print("Neraium intelligence layer — quick demo\n")
    with tempfile.TemporaryDirectory() as d:
        engine = StructuralEngine(
            baseline_window=25,
            recent_window=10,
            regime_store_path=f"{d}/r.json",
        )
        for t in range(45):
            base = math.sin(0.05 * t)
            frame = {
                "timestamp": str(t),
                "site_id": "demo",
                "asset_id": "asset-1",
                "sensor_values": {"s1": base, "s2": 0.95 * base, "s3": 1.0 * base},
            }
            out = engine.process_frame(frame)

        print("Sample output (last frame):")
        print("  interpreted_state:", out.get("interpreted_state"))
        print("  structural_drift_score:", out.get("structural_drift_score"))
        print("  latest_instability:", out.get("latest_instability"))
        print("  confidence_score:", out.get("confidence_score"))
        print("  baseline_mode:", out.get("baseline_mode"))
        print("  causal_attribution.top_drivers:", out.get("causal_attribution", {}).get("top_drivers"))
        print("  dominant_driver:", out.get("dominant_driver"))
        print("  data_quality_summary.gate_passed:", out.get("data_quality_summary", {}).get("gate_passed"))
        print("  active_sensor_count:", out.get("active_sensor_count"))
        print("  regime_memory_state:", out.get("regime_memory_state"))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
