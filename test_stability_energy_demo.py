#!/usr/bin/env python3
"""
Demo: Stability Energy Layer output for synthetic FD004-like data.

Shows the stability_energy fields per frame cycle.
"""

from __future__ import annotations

import json
import numpy as np
from neraium_core.engine.unified import Engine
from neraium_core.engine.schemas import InputFrame


def generate_fd004_like_frame(
    cycle: int,
    unit_id: str,
    baseline_mean: np.ndarray | None = None,
    degradation_factor: float = 1.0,
) -> InputFrame:
    """Generate a synthetic FD004-like frame."""
    # Simulate sensor data with degradation over time
    if baseline_mean is None:
        baseline_mean = np.array(
            [
                518.67, 643.02, 1589.7, 1400.0, 14.62, 21.61, 554.36, 2388.06,
                9046.19, 373.00, 2388.02, 8723.45, 8900.0, 160.0, 392.0, 2388.0,
                100.0, 393.0, 2388.0, 100.0, 393.0, 100.0, 393.0, 100.0
            ]
        )

    # Add noise and degradation
    noise = np.random.randn(24) * 5.0
    degradation = np.linspace(0, degradation_factor * 50.0, 24)
    sensors = baseline_mean + noise + degradation * (cycle / 100.0)

    return InputFrame(
        timestamp=float(cycle),
        unit_id=unit_id,
        sensors={f"s{i+1}": float(v) for i, v in enumerate(sensors)}
    )


def main():
    """Run demo and print stability_energy output."""
    print("=" * 80)
    print("STABILITY ENERGY LAYER DEMO: Synthetic FD004-like Data")
    print("=" * 80)
    print()

    engine = Engine()

    print("Ingesting baseline window (50 frames)...")
    for cycle in range(1, 51):
        frame = generate_fd004_like_frame(cycle, "unit_001")
        result = engine.process_frame(frame)
        if cycle % 10 == 0:
            print(f"  Cycle {cycle}: baseline_ready={result.baseline_ready}")

    print()
    print("Processing frames 51-70 (stable operation)...")
    for cycle in range(51, 71):
        frame = generate_fd004_like_frame(cycle, "unit_001")
        result = engine.process_frame(frame)

        # Extract and display stability_energy
        raw = engine._structural_engine.latest_result or {}
        se = raw.get("stability_energy", {})

        if cycle % 5 == 0:
            print(f"\nCycle {cycle} (Stable Phase):")
            print(f"  Energy:                    {se.get('energy', 0.0):.6f}")
            print(f"  Energy Δ:                   {se.get('energy_delta', 0.0):.6f}")
            print(f"  Energy Velocity:           {se.get('energy_velocity', 0.0):.6f}")
            print(f"  Energy Acceleration:       {se.get('energy_acceleration', 0.0):.6f}")
            print(f"  Instability Gradient Norm: {se.get('instability_gradient_norm', 0.0):.6f}")
            print(f"  Recovery Force Norm:       {se.get('recovery_force_norm', 0.0):.6f}")
            print(f"  Escape Alignment:          {se.get('escape_alignment', 0.0):.6f}")
            print(f"  Recovery Alignment:        {se.get('recovery_alignment', 0.0):.6f}")
            print(f"  Direction Label:           {se.get('direction_label', 'INSUFFICIENT_HISTORY')}")
            print(f"  Interpretation:            {se.get('interpretation', '')}")

    print()
    print("=" * 80)
    print("Processing frames 71-100 (degradation phase)...")
    for cycle in range(71, 101):
        frame = generate_fd004_like_frame(cycle, "unit_001", degradation_factor=2.0)
        result = engine.process_frame(frame)

        raw = engine._structural_engine.latest_result or {}
        se = raw.get("stability_energy", {})

        if cycle % 5 == 0:
            print(f"\nCycle {cycle} (Degradation Phase):")
            print(f"  Energy:                    {se.get('energy', 0.0):.6f}")
            print(f"  Energy Δ:                   {se.get('energy_delta', 0.0):.6f}")
            print(f"  Energy Velocity:           {se.get('energy_velocity', 0.0):.6f}")
            print(f"  Energy Acceleration:       {se.get('energy_acceleration', 0.0):.6f}")
            print(f"  Instability Gradient Norm: {se.get('instability_gradient_norm', 0.0):.6f}")
            print(f"  Recovery Force Norm:       {se.get('recovery_force_norm', 0.0):.6f}")
            print(f"  Escape Alignment:          {se.get('escape_alignment', 0.0):.6f}")
            print(f"  Recovery Alignment:        {se.get('recovery_alignment', 0.0):.6f}")
            print(f"  Direction Label:           {se.get('direction_label', 'INSUFFICIENT_HISTORY')}")
            print(f"  Interpretation:            {se.get('interpretation', '')}")

    print()
    print("=" * 80)
    print("Demo complete! Stability energy layer is operational.")
    print("=" * 80)


if __name__ == "__main__":
    main()
