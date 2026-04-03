"""Thin API for streaming / polling integration."""

from __future__ import annotations

from typing import Any

from neraium_core.realtime.update_engine import IncrementalUpdateEngine


def process_one(
    engine: IncrementalUpdateEngine,
    payload: dict[str, Any],
    *,
    attach_timing: bool = False,
) -> dict[str, Any]:
    out = engine.update(payload)
    if attach_timing and engine.state.timer.totals:
        out = dict(out)
        out["_incremental_timing"] = engine.state.timer.summary()
    return out
