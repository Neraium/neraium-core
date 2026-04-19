"""
Compatibility wrapper for demo/presentation code.

The actual demo implementation lives under `examples/demo/` to keep `neraium_core/`
focused on production/pipeline logic.
"""

from __future__ import annotations

from examples.demo.demo import (  # type: ignore
    SimulationConfig,
    _display_instability,
    _patch_engine_compatibility,
    _risk_and_message,
    _trend_arrow,
    generate_sensor_stream,
    run_demo,
)

__all__ = [
    "SimulationConfig",
    "_display_instability",
    "_patch_engine_compatibility",
    "_risk_and_message",
    "_trend_arrow",
    "generate_sensor_stream",
    "run_demo",
]

