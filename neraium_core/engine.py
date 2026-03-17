from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional


class StructuralEngine:
    """Runtime structural monitoring engine for Neraium."""

    def __init__(
        self,
        baseline_window: int = 24,
        recent_window: int = 8,
        max_frames: int = 500,
    ):
        if baseline_window < 2:
            raise ValueError("baseline_window must be at least 2")
        if recent_window < 2:
            raise ValueError("recent_window must be at least 2")
        if max_frames < (baseline_window + recent_window):
            raise ValueError("max_frames is too small for configured windows")

        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.frames: Deque[Dict] = deque(maxlen=max_frames)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.prev_drift: Optional[float] = None

    def _vector_from_frame(self, frame: Dict) -> Dict[str, float]:
        sensor_values = frame.get("sensor_values", {})
        if not isinstance(sensor_values, dict):
            raise ValueError("frame['sensor_values'] must be a dictionary")

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        vector: Dict[str, float] = {}
        for name in self.sensor_order:
            try:
                vector[name] = float(sensor_values.get(name))
            except (TypeError, ValueError):
                vector[name] = 0.0

        return vector

    def _mean_for(self, rows: List[Dict[str, float]], sensor: str) -> float:
        values = [row[sensor] for row in rows]
        return sum(values) / len(values)

    def process_frame(self, frame: Dict) -> Dict:
        required = {"timestamp", "site_id", "asset_id", "sensor_values"}
        missing = [key for key in required if key not in frame]
        if missing:
            raise ValueError(f"Frame is missing required keys: {missing}")

        vector = self._vector_from_frame(frame)
        self.frames.append({**frame, "_vector": vector})

        if len(self.frames) < (self.baseline_window + 1):
            result = {
                "state": "STABLE",
                "structural_drift_score": 0.0,
                "relational_stability_score": 1.0,
                "drift_velocity": 0.0,
            }
            self.latest_result = result
            return result

        rows = [item["_vector"] for item in self.frames]
        baseline_rows = rows[: self.baseline_window]

        deltas: List[float] = []
        for sensor in self.sensor_order:
            baseline_mean = self._mean_for(baseline_rows, sensor)
            deltas.append(abs(vector[sensor] - baseline_mean))

        drift_score = sum(deltas) / max(1, len(deltas))
        stability = max(0.0, 1.0 - drift_score / 10.0)

        velocity = 0.0 if self.prev_drift is None else drift_score - self.prev_drift
        self.prev_drift = drift_score

        state = "ALERT" if drift_score > 3 else "WATCH" if drift_score > 1.5 else "STABLE"

        result = {
            "state": state,
            "structural_drift_score": round(drift_score, 4),
            "relational_stability_score": round(stability, 4),
            "drift_velocity": round(velocity, 4),
        }
        self.latest_result = result
        return result

    def reset(self) -> None:
        self.frames.clear()
        self.prev_drift = None
        self.latest_result = None
        self.sensor_order = []
