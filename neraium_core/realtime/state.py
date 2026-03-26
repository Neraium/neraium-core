"""Persistent per-unit incremental state for structural updates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neraium_core.realtime.buffer import TimestampDequeBuffer, VectorDequeBuffer
from neraium_core.realtime.profiling import StageTimer


@dataclass
class NeraiumIncrementalState:
    """
    Per-asset buffers and lightweight summaries for incremental operation.
    Heavy structural logic remains in StructuralEngine; this holds rolling views.
    """

    unit_id: str
    baseline_window: int
    recent_window: int
    sensor_order: list[str] = field(default_factory=list)
    recent_vectors: VectorDequeBuffer | None = None
    recent_ts: TimestampDequeBuffer | None = None
    baseline_matrix: np.ndarray | None = None
    baseline_ts: list[float] | None = None
    baseline_frozen: bool = False
    score_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    timer: StageTimer = field(default_factory=StageTimer)
    last_result: dict[str, Any] | None = None
    graph_summary_cache: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.recent_vectors = VectorDequeBuffer(self.recent_window)
        self.recent_ts = TimestampDequeBuffer(self.recent_window)

    def reset_windows(self) -> None:
        self.baseline_matrix = None
        self.baseline_ts = None
        self.baseline_frozen = False
        if self.recent_vectors:
            self.recent_vectors.clear()
        if self.recent_ts:
            self.recent_ts.clear()
