import math
from collections import deque
from typing import Dict, List, Optional

import numpy as np


class StructuralEngine:
    """
    Runtime structural monitoring engine for Neraium.

    This engine:
    - converts normalized telemetry frames into ordered numeric vectors
    - builds a healthy baseline from early frames
    - computes Mahalanobis drift from the baseline manifold
    - computes covariance / relational drift between baseline and recent windows
    - combines both into a structural drift score
    - tracks drift velocity for early instability estimation
    - emits a pilot-friendly event/result object for each processed frame
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
    ):
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

        # Normalize weights defensively.
        self.mahal_weight = mahal_weight / total_weight
        self.cov_weight = cov_weight / total_weight

        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.smoothing_window = smoothing_window
        self.enable_vector_smoothing = enable_vector_smoothing

        self.frames = deque(maxlen=max_frames)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.prev_drift: Optional[float] = None

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        """
        Convert the frame's sensor dictionary into a consistent ordered vector.
        Missing or invalid values are represented as NaN and handled downstream.
        """
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

    def _valid_matrix(self, frames: List[Dict]) -> Optional[np.ndarray]:
        """
        Build a matrix from stored frame vectors, dropping rows with any NaN values.
        """
        if not frames:
            return None

        X = np.vstack([f["_vector"] for f in frames])

        # Drop incomplete rows so missing telemetry does not become fake zeros.
        row_mask = ~np.isnan(X).any(axis=1)
        X = X[row_mask]

        if len(X) < 2:
            return None

        return X

    def _smooth_matrix(self, X: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Apply light rolling smoothing across time for each sensor column.
        """
        if X is None or len(X) < 2 or self.smoothing_window <= 1:
            return X

        smoothed = X.copy()
        window = min(self.smoothing_window, len(X))
        kernel = np.ones(window, dtype=float) / float(window)

        for col in range(smoothed.shape[1]):
            smoothed[:, col] = np.convolve(smoothed[:, col], kernel, mode="same")

        return smoothed

    def _smooth_current_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Very light online smoothing against the previous vector to reduce spikes.
        """
        if not self.enable_vector_smoothing:
            return vector

        if len(self.frames) == 0:
            return vector

        prev = self.frames[-1]["_vector"]

        if prev.shape != vector.shape:
            return vector

        if np.isnan(prev).any() or np.isnan(vector).any():
            return vector

        return 0.5 * (vector + prev)

    def _regularize_covariance(self, cov: np.ndarray) -> np.ndarray:
        """
        Stabilize covariance via symmetry enforcement and eigenvalue flooring.
        """
        if cov.ndim == 0:
            cov = np.array([[float(cov)]], dtype=float)

        cov = 0.5 * (cov + cov.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)

        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _baseline_stats(self) -> Optional[Dict]:
        """
        Compute mean/covariance statistics from the baseline window.
        """
        if len(self.frames) < max(6, self.baseline_window):
            return None

        baseline_frames = list(self.frames)[: self.baseline_window]
        X = self._valid_matrix(baseline_frames)
        X = self._smooth_matrix(X)

        if X is None or len(X) < 2:
            return None

        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        cov = self._regularize_covariance(cov)
        inv_cov = np.linalg.inv(cov)

        return {
            "mean": mean,
            "cov": cov,
            "inv_cov": inv_cov,
        }

    def _mahalanobis(self, x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        """
        Mahalanobis distance from current state to baseline manifold.
        """
        delta = x - mean
        raw = float(delta.T @ inv_cov @ delta)
        raw = max(raw, 0.0)
        return float(math.sqrt(raw))

    def _covariance_drift(self) -> float:
        """
        Measure relational / structural drift between baseline and recent windows.
        """
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 0.0

        baseline_frames = list(self.frames)[: self.baseline_window]
        recent_frames = list(self.frames)[-self.recent_window :]

        Xb = self._valid_matrix(baseline_frames)
        Xr = self._valid_matrix(recent_frames)

        Xb = self._smooth_matrix(Xb)
        Xr = self._smooth_matrix(Xr)

        if Xb is None or Xr is None:
            return 0.0

        cov_b = np.cov(Xb, rowvar=False)
        cov_r = np.cov(Xr, rowvar=False)

        cov_b = self._regularize_covariance(cov_b)
        cov_r = self._regularize_covariance(cov_r)

        return float(np.linalg.norm(cov_r - cov_b, ord="fro"))

    def _relational_stability(self, cov_drift: float) -> float:
        """
        Convert covariance drift into a bounded stability score.
        Higher drift means lower stability.
        """
        score = 1.0 / (1.0 + cov_drift)
        return max(0.0, min(1.0, float(score)))

    def _combined_drift(self, mahal: float, cov_drift: float) -> float:
        """
        Fuse manifold departure and structural deformation into one drift score.
        """
        return (self.mahal_weight * mahal) + (self.cov_weight * cov_drift)

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        """
        Convert drift + stability into an operator-facing 0-100 health score.
        """
        health = 100.0 - min(drift_score * 18.0, 80.0)
        health += stability_score * 20.0
        health = max(0.0, min(100.0, health))
        return int(round(health))

    def _alert_state(self, drift_score: float) -> str:
        """
        Convert drift score into a simple alert state.
        """
        if drift_score > 3.0:
            return "ALERT"
        if drift_score > 1.5:
            return "WATCH"
        return "STABLE"

    def process_frame(self, frame: Dict) -> Dict:
        """
        Process one normalized telemetry frame and return the latest structural event.
        Expected frame shape:
        {
            "timestamp": "...",
            "site_id": "...",
            "asset_id": "...",
            "sensor_values": {...}
        }
        """
        required = {"timestamp", "site_id", "asset_id", "sensor_values"}
        missing = [key for key in required if key not in frame]
        if missing:
            raise ValueError(f"Frame is missing required keys: {missing}")

        vector = self._vector_from_frame(frame)
        vector = self._smooth_current_vector(vector)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        baseline = self._baseline_stats()

        # Wait for enough healthy baseline or skip scoring incomplete current frames.
        if baseline is None or np.isnan(vector).any():
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
                "mahalanobis_score": 0.0,
                "covariance_drift_score": 0.0,
            }
            self.latest_result = result
            return result

        mahal = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])
        cov_drift = self._covariance_drift()
        drift_score = self._combined_drift(mahal, cov_drift)

        stability_score = self._relational_stability(cov_drift)
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
            "mahalanobis_score": round(mahal, 4),
            "covariance_drift_score": round(cov_drift, 4),
        }

        self.latest_result = result
        return result


if __name__ == "__main__":
    # Minimal smoke test / example usage.
    engine = StructuralEngine()

    sample_frames = [
        {
            "timestamp": f"2026-03-13T00:{i:02d}:00Z",
            "site_id": "site-a",
            "asset_id": "asset-1",
            "sensor_values": {
                "temp": 50.0 + (i * 0.02),
                "pressure": 100.0 + (i * 0.03),
                "vibration": 1.0 + (i * 0.005),
            },
        }
        for i in range(40)
    ]

    for frame in sample_frames:
        output = engine.process_frame(frame)

    print(output)