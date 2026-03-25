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
        self._transition_pressure_history: deque[float] = deque(maxlen=60)
        self._shock_activity_history: deque[float] = deque(maxlen=60)
        self._structural_drift_history: deque[float] = deque(maxlen=60)
        self._state_history: deque[str] = deque(maxlen=60)
        self._frame_count: int = 0

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

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _trend(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(values[-1] - values[0]) / float(len(values) - 1)

    def _update_histories(self, *, drift_score: float, velocity: float, state: str, pressure: float) -> None:
        shock_activity = self._clamp01(abs(velocity) / 0.35)
        self._transition_pressure_history.append(float(pressure))
        self._shock_activity_history.append(float(shock_activity))
        self._structural_drift_history.append(float(drift_score))
        self._state_history.append(state)

    def _compute_base_metrics(
        self,
        frame: Dict,
        baseline: Optional[Dict],
        vector: np.ndarray,
    ) -> Dict:
        base = {
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
            "latest_instability": 0.0,
            "confidence_score": 0.25,
            "interpreted_state": "BASELINE_FORMING",
            "regime_name": "INITIALIZING",
        }

        if baseline is None or np.isnan(vector).any():
            return base

        mahal = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])
        cov_drift = self._covariance_drift()
        drift_score = self._combined_drift(mahal, cov_drift)
        stability_score = self._relational_stability(cov_drift)
        health = self._system_health(drift_score, stability_score)
        state = self._alert_state(drift_score)
        drift_alert = drift_score > 1.5

        velocity = 0.0 if self.prev_drift is None else (drift_score - self.prev_drift)
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

        interpreted_state = "STABLE_FLOW"
        regime_name = "NOMINAL"
        if state == "WATCH":
            interpreted_state = "EMERGING_TRANSITION"
            regime_name = "DEGRADING"
        elif state == "ALERT":
            interpreted_state = "SUSTAINED_TRANSITION"
            regime_name = "INSTABILITY_REGIME"

        return {
            **base,
            "event_type": event_type,
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
            "latest_instability": round(self._clamp01(drift_score / 4.5), 4),
            "confidence_score": round(self._clamp01(0.2 + 0.8 * stability_score), 4),
            "interpreted_state": interpreted_state,
            "regime_name": regime_name,
        }

    def _compute_transition_metrics(self, *, drift_score: float, velocity: float, stability_score: float) -> Dict:
        raw_pressure = (0.65 * self._clamp01(drift_score / 3.0)) + (0.25 * self._clamp01(velocity / 0.25)) + (
            0.10 * self._clamp01((1.0 - stability_score) / 0.75)
        )
        pressure = self._clamp01(raw_pressure)
        if pressure >= 0.75:
            transition_state = "SUSTAINED_TRANSITION"
        elif pressure >= 0.45:
            transition_state = "EMERGING_TRANSITION"
        elif pressure >= 0.2:
            transition_state = "METASTABLE"
        else:
            transition_state = "NONE"
        return {"transition_pressure": round(pressure, 4), "transition_state": transition_state}

    def _analytics_unavailable(self, reason: str) -> Dict:
        return {"available": False, "reason": reason}

    def _compute_advanced_analytics(self) -> Dict:
        if len(self._structural_drift_history) < 3:
            unavailable = self._analytics_unavailable("insufficient history")
            return {
                "trajectory_analysis": unavailable,
                "branching_analysis": unavailable,
                "constraint_analysis": unavailable,
                "hierarchy_analysis": unavailable,
                "horizon_analysis": unavailable,
                "counterfactual_simulation": unavailable,
            }

        pressure_tail = list(self._transition_pressure_history)[-6:]
        drift_tail = list(self._structural_drift_history)[-6:]
        shock_tail = list(self._shock_activity_history)[-6:]

        pressure_trend = self._trend(pressure_tail)
        drift_trend = self._trend(drift_tail)
        avg_pressure = float(np.mean(pressure_tail)) if pressure_tail else 0.0
        avg_shock = float(np.mean(shock_tail)) if shock_tail else 0.0

        if pressure_trend < -0.03 and drift_trend < -0.03:
            dominant_path = "STABILIZING"
        elif pressure_trend > 0.02 or drift_trend > 0.03:
            dominant_path = "DIVERGING"
        else:
            dominant_path = "METASTABLE"

        path_confidence = self._clamp01(0.35 + abs(pressure_trend) * 4.0 + abs(drift_trend) * 2.5)
        trajectory = {
            "available": True,
            "dominant_path": dominant_path,
            "path_confidence": round(path_confidence, 4),
            "trend_strength": round(abs(drift_trend) + abs(pressure_trend), 4),
        }

        decision_tension = self._clamp01((avg_shock * 0.5) + (abs(pressure_trend) * 4.0))
        commitment = self._clamp01(1.0 - decision_tension)
        branching = {
            "available": True,
            "is_branching": bool(decision_tension >= 0.45 and dominant_path == "METASTABLE"),
            "commitment": round(commitment, 4),
            "decision_tension": round(decision_tension, 4),
        }

        lock_in_score = self._clamp01((avg_pressure * 0.6) + (float(np.mean(drift_tail)) / 4.5) * 0.4)
        constraint = {
            "available": True,
            "lock_in_score": round(lock_in_score, 4),
            "point_of_no_return_risk": round(self._clamp01(lock_in_score * 1.15), 4),
            "recovery_margin": round(self._clamp01(1.0 - lock_in_score), 4),
        }

        origin_scope = "LOCAL"
        if lock_in_score > 0.7:
            origin_scope = "SYSTEMIC"
        elif lock_in_score > 0.45:
            origin_scope = "MULTI_SUBSYSTEM"
        hierarchy = {
            "available": True,
            "origin_scope": origin_scope,
            "propagation_risk": round(self._clamp01((lock_in_score * 0.7) + (avg_shock * 0.3)), 4),
        }

        horizon_bucket = "LONGER_HORIZON"
        horizon_signal = (avg_pressure * 0.5) + (lock_in_score * 0.5)
        if dominant_path == "DIVERGING":
            horizon_signal += 0.2
        if horizon_signal >= 0.85:
            horizon_bucket = "IMMINENT"
        elif horizon_signal >= 0.65:
            horizon_bucket = "NEAR_TERM"
        elif horizon_signal >= 0.4:
            horizon_bucket = "MID_TERM"
        horizon = {"available": True, "risk_horizon": horizon_bucket, "horizon_score": round(self._clamp01(horizon_signal), 4)}

        baseline_path = "DIVERGING" if dominant_path == "DIVERGING" else "METASTABLE"
        counterfactual = {
            "available": True,
            "baseline_future": {
                "projected_path": baseline_path,
                "projected_lock_in_risk": round(lock_in_score, 4),
                "projected_horizon": horizon_bucket,
            },
            "counterfactuals": [
                {"scenario": "pressure_relief", "projected_path": "STABILIZING"},
                {"scenario": "status_quo", "projected_path": baseline_path},
                {"scenario": "continued_degradation", "projected_path": "DIVERGING"},
            ],
        }

        return {
            "trajectory_analysis": trajectory,
            "branching_analysis": branching,
            "constraint_analysis": constraint,
            "hierarchy_analysis": hierarchy,
            "horizon_analysis": horizon,
            "counterfactual_simulation": counterfactual,
        }

    def _assemble_result(self, base: Dict, transition: Dict, analytics: Dict) -> Dict:
        result = {
            **base,
            "transition_pressure": transition["transition_pressure"],
            "transition_state": transition["transition_state"],
            "experimental_analytics": analytics,
        }
        return result

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
        base = self._compute_base_metrics(frame=frame, baseline=baseline, vector=vector)
        transition = self._compute_transition_metrics(
            drift_score=float(base["structural_drift_score"]),
            velocity=float(base["drift_velocity"]),
            stability_score=float(base["relational_stability_score"]),
        )
        self._update_histories(
            drift_score=float(base["structural_drift_score"]),
            velocity=float(base["drift_velocity"]),
            state=str(base["state"]),
            pressure=float(transition["transition_pressure"]),
        )
        analytics = self._compute_advanced_analytics()
        result = self._assemble_result(base=base, transition=transition, analytics=analytics)

        self._frame_count += 1
        if self._frame_count <= 3:
            print("DEBUG RESULT KEYS:", result.keys())
            print("DEBUG EXPERIMENTAL:", result["experimental_analytics"])
            print(
                "DEBUG ADVANCED POPULATED:",
                {name: bool(isinstance(payload, dict) and payload.get("available")) for name, payload in result["experimental_analytics"].items()},
            )

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
