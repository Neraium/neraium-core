"""Realtime-friendly batching over observations."""

from __future__ import annotations

from typing import Any, Iterator

from neraium_core.realtime.update_engine import IncrementalUpdateEngine


class RealtimeRunner:
    """Feeds ordered observations to one or more IncrementalUpdateEngine instances."""

    def __init__(self) -> None:
        self._units: dict[str, IncrementalUpdateEngine] = {}

    def engine_for(
        self,
        unit_id: str,
        *,
        baseline_window: int = 50,
        recent_window: int = 12,
    ) -> IncrementalUpdateEngine:
        if unit_id not in self._units:
            self._units[unit_id] = IncrementalUpdateEngine(
                unit_id,
                baseline_window=baseline_window,
                recent_window=recent_window,
            )
        return self._units[unit_id]

    def process_stream(self, frames: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        for frame in frames:
            asset = str(frame.get("asset_id", "default"))
            eng = self.engine_for(asset)
            yield eng.update(frame)
