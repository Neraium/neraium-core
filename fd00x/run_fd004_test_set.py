#!/usr/bin/env python3
r"""Convenience runner for FD004 test set at C:\Users\Owner\Desktop\CMAPSSData\test_FD004.

This script automatically points to the test dataset and runs advanced analysis.

Usage:
    python run_fd004_test_set.py [--max-units N] [--output-dir DIR] [--basic]

Examples:
    # Run full advanced analysis on test set
    python run_fd004_test_set.py

    # Run on first 50 units
    python run_fd004_test_set.py --max-units 50

    # Use basic runner instead
    python run_fd004_test_set.py --basic

    # Custom output directory
    python run_fd004_test_set.py --output-dir ./my_test_results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .run_drift_detection_demo import DriftDetectionDemoRunner
from .run_advanced_fd004_demo import AdvancedFD004Demo


def find_fd004_test_set() -> str:
    r"""Locate FD004 test set.

    Searches in common locations:
    1. C:\Users\Owner\Desktop\CMAPSSData\test_FD004.txt
    2. ./data/test_FD004.txt
    3. ./test_FD004.txt
    4. Relative to script
    """
    candidates = [
        r"C:\Users\Owner\Desktop\CMAPSSData\test_FD004.txt",
        "data/test_FD004.txt",
        "test_FD004.txt",
        Path(__file__).parent.parent / "data" / "test_FD004.txt",
    ]

    for path_str in candidates:
        path = Path(path_str)
        if path.exists():
            return str(path.resolve())

    raise FileNotFoundError(
        f"Could not find FD004 test set. Searched:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + f"\n\nPlease provide --path argument or place test_FD004.txt in one of these locations."
    )


def main():
    """Run drift detection on FD004 test set."""
    parser = argparse.ArgumentParser(
        description="Drift detection for FD004 test set (automatic path detection)"
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Override FD004 dataset path (auto-detects if not provided)",
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Limit to first N units",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fd004_test_set_results",
        help="Output directory (default: fd004_test_set_results)",
    )
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use basic runner instead of advanced (advanced recommended)",
    )

    args = parser.parse_args()

    # Find dataset path
    if args.path:
        dataset_path = args.path
        if not Path(dataset_path).exists():
            print(f"Error: Dataset file not found: {dataset_path}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            dataset_path = find_fd004_test_set()
            print(f"✓ Found FD004 test set: {dataset_path}\n")
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Run appropriate runner
    try:
        if args.basic:
            print("Using BASIC runner...\n")
            demo = DriftDetectionDemoRunner(output_dir=args.output_dir, verbose=True)
            demo.run(
                dataset_path=dataset_path,
                max_units=args.max_units,
                skip_plots=False,
            )
        else:
            print("Using ADVANCED runner (recommended for FD004)...\n")
            demo = AdvancedFD004Demo(output_dir=args.output_dir, verbose=True)
            demo.run(
                dataset_path=dataset_path,
                max_units=args.max_units,
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
