"""Orchestrates ingestion → structural update → output (batch-compatible)."""

from __future__ import annotations

from typing import Any

from neraium_core.alignment import StructuralEngine
from neraium_core.realtime.profiling import profiling_enabled, run_timed
from neraium_core.realtime.state import NeraiumIncrementalState


class IncrementalUpdateEngine:
    """
    Per-unit facade over StructuralEngine with explicit lifecycle.

    Structural semantics are unchanged; the engine uses incremental window buffers
    when ``NERAIUM_INCREMENTAL`` is enabled (default: on).
    """

    def __init__(
        self,
        unit_id: str,
        *,
        baseline_window: int = 50,
        recent_window: int = 12,
        window_stride: int = 1,
    ) -> None:
        self.unit_id = unit_id
        self.state = NeraiumIncrementalState(
            unit_id=unit_id,
            baseline_window=baseline_window,
            recent_window=recent_window,
        )
        self._engine = StructuralEngine(
            baseline_window=baseline_window,
            recent_window=recent_window,
            window_stride=window_stride,
        )

    def initialize(self) -> None:
        """Reset engine and per-unit state."""
        self._engine = StructuralEngine(
            baseline_window=self.state.baseline_window,
            recent_window=self.state.recent_window,
        )
        self.state.reset_windows()
        self.state.last_result = None

    def update(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Process one telemetry frame; returns the same payload as batch ``process_frame``."""
        frame = dict(observation)
        if profiling_enabled():
            out = run_timed("structural_update", self.state.timer, lambda: self._engine.process_frame(frame))
        else:
            out = self._engine.process_frame(frame)
        self.state.last_result = out
        return out

    def emit_current_result(self) -> dict[str, Any] | None:
        return self.state.last_result

    @property
    def engine(self) -> StructuralEngine:
        return self._engine
