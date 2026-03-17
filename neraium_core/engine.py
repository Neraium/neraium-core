from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List

from neraium_core.ingest import normalize_telemetry_frame
from neraium_core.scoring import compute_drift_score


class StructuralEngine:
    def __init__(self, baseline_window: int = 100, recent_window: int = 20):
        if baseline_window < 2:
            raise ValueError("baseline_window must be >= 2")
        if recent_window < 2:
            raise ValueError("recent_window must be >= 2")
        if recent_window >= baseline_window:
            raise ValueError("recent_window must be smaller than baseline_window")

        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self._frames: Deque[list[float]] = deque(maxlen=baseline_window + recent_window)
        self._sensor_order: List[str] = []

    def _frame_to_vector(self, frame: Dict) -> list[float]:
        sensor_values = frame["sensor_values"]
        if not self._sensor_order:
            self._sensor_order = sorted(sensor_values.keys())
        return [float(sensor_values.get(name, 0.0)) for name in self._sensor_order]

    def ingest(self, frame: Dict):
        """Add telemetry frame"""
        normalized = normalize_telemetry_frame(frame)
        self._frames.append(self._frame_to_vector(normalized))

    def score(self) -> float:
        """Return structural drift score"""
        if len(self._frames) < self.baseline_window:
            return 0.0

        all_frames = list(self._frames)
        baseline = all_frames[: self.baseline_window]
        recent = all_frames[-self.recent_window :]
        sample = recent[-1]

        return compute_drift_score(sample=sample, baseline=baseline, recent=recent)

    def reset(self):
        """Clear internal state"""
        self._frames.clear()
        self._sensor_order = []
