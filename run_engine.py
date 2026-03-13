import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np


class StructuralEngine:
    def __init__(self, baseline_window: int = 24, recent_window: int = 8):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.prev_drift: Optional[float] = None

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        sensor_values = frame["sensor_values"]

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values = []
        for name in self.sensor_order:
            v = sensor_values.get(name, 0.0)
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(0.0)

        return np.array(values, dtype=float)

    def _baseline_stats(self) -> Optional[Dict]:
        if len(self.frames) < max(6, self.baseline_window):
            return None

        baseline_frames = list(self.frames)[: self.baseline_window]
        X = np.vstack([f["_vector"] for f in baseline_frames])

        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)

        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)

        cov = cov + np.eye(cov.shape[0]) * 1e-6
        inv_cov = np.linalg.pinv(cov)

        return {
            "mean": mean,
            "inv_cov": inv_cov,
        }

    def _mahalanobis(self, x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        delta = x - mean
        return float(math.sqrt(delta.T @ inv_cov @ delta))

    def _relational_stability(self) -> float:
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 1.0

        baseline_frames = list(self.frames)[: self.baseline_window]
        recent_frames = list(self.frames)[-self.recent_window:]

        Xb = np.vstack([f["_vector"] for f in baseline_frames])
        Xr = np.vstack([f["_vector"] for f in recent_frames])

        cov_b = np.cov(Xb, rowvar=False)
        cov_r = np.cov(Xr, rowvar=False)

        if cov_b.ndim == 0:
            cov_b = np.array([[float(cov_b)]], dtype=float)
        if cov_r.ndim == 0:
            cov_r = np.array([[float(cov_r)]], dtype=float)

        diff = np.linalg.norm(cov_r - cov_b, ord="fro")
        score = 1.0 / (1.0 + diff)
        return max(0.0, min(1.0, float(score)))

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        health = 100.0 - min(drift_score * 18.0, 80.0)
        health += stability_score * 20.0
        health = max(0.0, min(100.0, health))
        return int(round(health))

    def _alert_state(self, drift_score: float) -> str:
        if drift_score > 3.0:
            return "ALERT"
        if drift_score > 1.5:
            return "WATCH"
        return "STABLE"

    def process_frame(self, frame: Dict) -> Dict:
        vector = self._vector_from_frame(frame)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        baseline = self._baseline_stats()

        if baseline is None:
            result = {
                "id": None,
                "event_type": "baseline_telemetry",
                "timestamp": frame["timestamp"],
                "site_id": frame["site_id"],
                "asset_id": frame["asset_id"],
                "state": "STABLE",
                "structural_drift_score": 0.0,
                "relational_stability_score": 1.0,
                "system_health": 100,
                "drift_alert": False,
                "lead_time_hours": None,
                "lead_time_confidence": 0.0,
                "drift_velocity": 0.0,
                "structural_driver": "baseline formation",
                "predicted_impact": "No near term operational disruption expected.",
                "explanation": "Initializing structural telemetry...",
            }
            self.latest_result = result
            return result

        drift_score = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])
        stability_score = self._relational_stability()
        health = self._system_health(drift_score, stability_score)
        state = self._alert_state(drift_score)
        drift_alert = drift_score > 1.5

        if self.prev_drift is None:
            velocity = 0.0
        else:
            velocity = drift_score - self.prev_drift
        self.prev_drift = drift_score

        lead_time_hours = None
        if velocity > 0.01:
            remaining = max(0.0, 4.5 - drift_score)
            lead_time_hours = round(min(240.0, remaining / velocity), 1)

        if state == "ALERT":
            event_type = "instability_escalation"
            impact = "Potential localized service disruption within 1 to 2 hours."
            driver = "cross-sensor structural divergence"
        elif state == "WATCH":
            event_type = "quality_observation"
            impact = "Early degradation detected. Maintenance window recommended."
            driver = "emerging relational drift"
        else:
            event_type = "flow_observation"
            impact = "No near term operational disruption expected."
            driver = "stable baseline telemetry"

        result = {
            "id": None,
            "event_type": event_type,
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": state,
            "structural_drift_score": round(drift_score, 4),
            "relational_stability_score": round(stability_score, 4),
            "system_health": health,
            "drift_alert": drift_alert,
            "lead_time_hours": lead_time_hours,
            "lead_time_confidence": round(min(0.99, max(0.35, stability_score)), 2),
            "drift_velocity": round(velocity, 4),
            "structural_driver": driver,
            "predicted_impact": impact,
            "explanation": "Structural engine monitoring sensor relationships in real time.",
        }

        self.latest_result = result
        return result