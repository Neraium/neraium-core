"""
Incremental / near-real-time structural update path.

Heavy imports (``IncrementalUpdateEngine``, etc.) are lazy to avoid circular imports
with ``neraium_core.alignment``.

- Enable profiling with env ``NERAIUM_PROFILE_INCREMENTAL=1``.
- Disable incremental window caching with ``NERAIUM_INCREMENTAL=0``.
"""

from __future__ import annotations

from typing import Any

from neraium_core.realtime.buffer import TimestampDequeBuffer, VectorDequeBuffer
from neraium_core.realtime.profiling import StageTimer, profiling_enabled
from neraium_core.realtime.state import NeraiumIncrementalState

__all__ = [
    "IncrementalUpdateEngine",
    "NeraiumIncrementalState",
    "RealtimeRunner",
    "StageTimer",
    "TimestampDequeBuffer",
    "VectorDequeBuffer",
    "process_one",
    "profiling_enabled",
]


def __getattr__(name: str) -> Any:
    if name == "IncrementalUpdateEngine":
        from neraium_core.realtime.update_engine import IncrementalUpdateEngine

        return IncrementalUpdateEngine
    if name == "RealtimeRunner":
        from neraium_core.realtime.runner import RealtimeRunner

        return RealtimeRunner
    if name == "process_one":
        from neraium_core.realtime.api import process_one

        return process_one
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
