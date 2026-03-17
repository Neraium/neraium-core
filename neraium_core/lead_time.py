from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class DetectorConfig:
    baseline_window: int = 120
    live_window: int = 24
    ema_alpha: float = 0.20
    ridge: float = 1e-4
    instability_boundary: float = 4.5
    min_velocity: float = 1e-6
    warning_cap_hours: int = 240


@dataclass
class DetectionResult:
    site_id: str
    asset_id: str
    timestamp: str
    structural_drift_score: float
    smoothed_drift_score: float
    drift_velocity: float
    drift_acceleration: float
    relational_stability_score: float
    lead_time_hours: Optional[float]
    lead_time_lower_hours: Optional[float]
    lead_time_upper_hours: Optional[float]
    lead_time_confidence: float
    state: str
    structural_driver: str


class HybridSIIDetector:
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._frames: Dict[Tuple[str, str], Deque[List[float]]] = defaultdict(
            lambda: deque(maxlen=self.config.baseline_window)
        )
        self._smoothed: Dict[Tuple[str, str], float] = {}
        self._velocity: Dict[Tuple[str, str], float] = {}

    def update(
        self,
        site_id: str,
        asset_id: str,
        timestamp: str,
        sensor_names: Tuple[str, ...],
        sensor_values: Tuple[Optional[float], ...],
        missing_fraction: float = 0.0,
    ) -> DetectionResult:
        key = (site_id, asset_id)
        x = [0.0 if (v is None or not math.isfinite(v)) else float(v) for v in sensor_values]
        history = self._frames[key]
        history.append(x)

        if len(history) < 8:
            return DetectionResult(
                site_id,
                asset_id,
                timestamp,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                None,
                None,
                None,
                0.0,
                "STABLE",
                "warmup",
            )

        baseline = history[0]
        raw_drift = math.sqrt(sum((x[i] - baseline[i]) ** 2 for i in range(len(x))))
        prev = self._smoothed.get(key, raw_drift)
        smoothed = self.config.ema_alpha * raw_drift + (1.0 - self.config.ema_alpha) * prev
        self._smoothed[key] = smoothed
        velocity = smoothed - prev
        acceleration = velocity - self._velocity.get(key, 0.0)
        self._velocity[key] = velocity

        lead = lower = upper = None
        if velocity > self.config.min_velocity:
            lead = max(
                0.0,
                min(
                    (self.config.instability_boundary - smoothed) / velocity,
                    float(self.config.warning_cap_hours),
                ),
            )
            lower = max(0.0, lead * 0.75)
            upper = min(float(self.config.warning_cap_hours), lead * 1.25)

        stability = 1.0 / (1.0 + abs(velocity))
        state = (
            "ALERT"
            if smoothed >= 3.5 or stability < 0.35
            else "WATCH"
            if smoothed >= 2.0 or stability < 0.60
            else "STABLE"
        )
        driver = (
            sensor_names[max(range(len(x)), key=lambda idx: abs(x[idx] - baseline[idx]))]
            if sensor_names
            else "unknown"
        )
        confidence = max(
            0.05, min(0.99, 1.0 - (missing_fraction * 0.75) - min(abs(acceleration) * 0.05, 0.2))
        )

        return DetectionResult(
            site_id=site_id,
            asset_id=asset_id,
            timestamp=timestamp,
            structural_drift_score=round(raw_drift, 4),
            smoothed_drift_score=round(smoothed, 4),
            drift_velocity=round(velocity, 4),
            drift_acceleration=round(acceleration, 4),
            relational_stability_score=round(stability, 4),
            lead_time_hours=None if lead is None else round(lead, 2),
            lead_time_lower_hours=None if lower is None else round(lower, 2),
            lead_time_upper_hours=None if upper is None else round(upper, 2),
            lead_time_confidence=round(confidence, 4),
            state=state,
            structural_driver=driver,
        )
