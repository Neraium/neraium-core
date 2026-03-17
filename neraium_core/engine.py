import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

Frame = Dict[str, Any]
Result = Dict[str, Any]


class StructuralEngine:
    """Structural analytics engine used by Neraium.

    Rigorous observables:
    - Mahalanobis distance from baseline manifold
    - covariance drift between baseline and recent windows
    - fused structural drift score and derived relational stability

    Heuristic/operator-facing outputs:
    - state buckets (STABLE/WATCH/ALERT)
    - system health scalar
    - drift velocity and lead-time estimate
    - textual impact/driver descriptions
    """

    def __init__(
        self,
        baseline_window: int = 24,
        recent_window: int = 8,
        max_frames: int = 500,
        mahal_weight: float = 0.65,
        cov_weight: float = 0.35,
        smoothing_window: int = 3,
        enable_vector_smoothing: bool = True,
    ) -> None:
        if baseline_window < 2:
            raise ValueError("baseline_window must be at least 2")
        if recent_window < 2:
            raise ValueError("recent_window must be at least 2")
        if max_frames < (baseline_window + recent_window):
            raise ValueError("max_frames is too small for configured windows")
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be at least 1")

        total_weight = mahal_weight + cov_weight
        if total_weight <= 0:
            raise ValueError("mahal_weight + cov_weight must be > 0")

        self.mahal_weight = mahal_weight / total_weight
        self.cov_weight = cov_weight / total_weight

        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.smoothing_window = smoothing_window
        self.enable_vector_smoothing = enable_vector_smoothing

        self.frames: Deque[Frame] = deque(maxlen=max_frames)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Result] = None
        self.prev_drift: Optional[float] = None

    def reset(self) -> None:
        """Reset all in-memory state for baseline rebuild and replay."""
        self.frames.clear()
        self.prev_drift = None
        self.latest_result = None
        self.sensor_order = []

    def get_latest_result(self) -> Optional[Result]:
        """Return the latest computed result if available."""
        return self.latest_result

    def _vector_from_frame(self, frame: Frame) -> np.ndarray:
        sensor_values = frame.get("sensor_values", {})
        if not isinstance(sensor_values, dict):
            raise ValueError("frame['sensor_values'] must be a dictionary")

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values: List[float] = []
        for name in self.sensor_order:
            value = sensor_values.get(name)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(np.nan)

        return np.array(values, dtype=float)

    def _valid_matrix(self, frames: List[Frame]) -> Optional[np.ndarray]:
        if not frames:
            return None

        matrix = np.vstack([f["_vector"] for f in frames])
        matrix = matrix[~np.isnan(matrix).any(axis=1)]
        if len(matrix) < 2:
            return None
        return matrix

    def _smooth_matrix(self, matrix: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if matrix is None or len(matrix) < 2 or self.smoothing_window <= 1:
            return matrix

        smoothed = matrix.copy()
        window = min(self.smoothing_window, len(matrix))
        kernel = np.ones(window, dtype=float) / float(window)
        for col in range(smoothed.shape[1]):
            smoothed[:, col] = np.convolve(smoothed[:, col], kernel, mode="same")
        return smoothed

    def _smooth_current_vector(self, vector: np.ndarray) -> np.ndarray:
        if not self.enable_vector_smoothing or len(self.frames) == 0:
            return vector

        prev = self.frames[-1]["_vector"]
        if prev.shape != vector.shape or np.isnan(prev).any() or np.isnan(vector).any():
            return vector
        return 0.5 * (vector + prev)

    def _regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _baseline_stats(self) -> Optional[Dict[str, np.ndarray]]:
        if len(self.frames) < max(6, self.baseline_window):
            return None

        baseline_frames = list(self.frames)[: self.baseline_window]
        matrix = self._smooth_matrix(self._valid_matrix(baseline_frames))
        if matrix is None or len(matrix) < 2:
            return None

        mean = np.mean(matrix, axis=0)
        cov = self._regularize_covariance(np.cov(matrix, rowvar=False))
        inv_cov = np.linalg.inv(cov)
        return {"mean": mean, "cov": cov, "inv_cov": inv_cov}

    def _mahalanobis(self, x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        delta = x - mean
        raw = float(delta.T @ inv_cov @ delta)
        return float(math.sqrt(max(raw, 0.0)))

    def _covariance_drift(self) -> float:
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 0.0

        baseline_frames = list(self.frames)[: self.baseline_window]
        recent_frames = list(self.frames)[-self.recent_window :]
        xb = self._smooth_matrix(self._valid_matrix(baseline_frames))
        xr = self._smooth_matrix(self._valid_matrix(recent_frames))
        if xb is None or xr is None:
            return 0.0

        cov_b = self._regularize_covariance(np.cov(xb, rowvar=False))
        cov_r = self._regularize_covariance(np.cov(xr, rowvar=False))
        return float(np.linalg.norm(cov_r - cov_b, ord="fro"))

    def process_frame(self, frame: Frame) -> Result:
        required = {"timestamp", "site_id", "asset_id", "sensor_values"}
        missing = [k for k in required if k not in frame]
        if missing:
            raise ValueError(f"Frame is missing required keys: {missing}")

        vector = self._smooth_current_vector(self._vector_from_frame(frame))
        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        baseline = self._baseline_stats()
        if baseline is None or np.isnan(vector).any():
            result: Result = {
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
                "mahalanobis_score": 0.0,
                "covariance_drift_score": 0.0,
            }
            self.latest_result = result
            return result

        mahal = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])
        cov_drift = self._covariance_drift()
        drift_score = (self.mahal_weight * mahal) + (self.cov_weight * cov_drift)
        stability = max(0.0, min(1.0, 1.0 / (1.0 + cov_drift)))

        health_value = 100.0 - min(drift_score * 18.0, 80.0) + (stability * 20.0)
        health = int(round(max(0.0, min(100.0, health_value))))
        if drift_score > 3.0:
            state = "ALERT"
        elif drift_score > 1.5:
            state = "WATCH"
        else:
            state = "STABLE"

        velocity = 0.0 if self.prev_drift is None else drift_score - self.prev_drift
        self.prev_drift = drift_score

        lead_time_hours: Optional[float] = None
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
            "relational_stability_score": round(stability, 4),
            "system_health": health,
            "drift_alert": drift_score > 1.5,
            "lead_time_hours": lead_time_hours,
            "lead_time_confidence": round(min(0.99, max(0.35, stability)), 2),
            "drift_velocity": round(velocity, 4),
            "structural_driver": driver,
            "predicted_impact": impact,
            "explanation": "Structural engine monitoring sensor relationships in real time.",
            "mahalanobis_score": round(mahal, 4),
            "covariance_drift_score": round(cov_drift, 4),
        }
        self.latest_result = result
        return result
