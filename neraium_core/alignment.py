import math
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from neraium_core.directional import directional_metrics, lagged_correlation_matrix
from neraium_core.early_warning import early_warning_metrics
from neraium_core.entropy import interaction_entropy
from neraium_core.geometry import correlation_matrix
from neraium_core.scoring import composite_instability_score
from neraium_core.spectral import spectral_gap, spectral_radius
from neraium_core.subsystems import subsystem_spectral_measures


class StructuralEngine:
    def __init__(self, baseline_window: int = 50, recent_window: int = 12):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None

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
        if len(self.frames) < max(5, self.baseline_window):
            return None

        baseline_frames = list(self.frames)[: self.baseline_window]
        X = np.vstack([f["_vector"] for f in baseline_frames])

        mean = X.mean(axis=0)

        if X.shape[1] == 0:
            cov = np.zeros((0, 0), dtype=float)
            inv_cov = np.zeros((0, 0), dtype=float)
        else:
            cov = np.cov(X, rowvar=False)

            if cov.ndim == 0:
                cov = np.array([[float(cov)]], dtype=float)

            cov = cov + np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.pinv(cov)

        return {
            "mean": mean,
            "cov": cov,
            "inv_cov": inv_cov,
        }

    def _mahalanobis(self, x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> float:
        delta = x - mean
        return float(math.sqrt(delta.T @ inv_cov @ delta))

    def _relational_stability(self) -> float:
        if len(self.frames) < max(self.baseline_window + self.recent_window, 10):
            return 1.0

        baseline_frames = list(self.frames)[: self.baseline_window]
        recent_frames = list(self.frames)[-self.recent_window :]

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

    def _drift_alert(self, drift_score: float) -> bool:
        return drift_score > 1.5

    def _count_valid_signals(self, frame: Dict[str, Any]) -> int:
        valid_signals = 0
        for value in frame.get("sensor_values", {}).values():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            if np.isfinite(numeric_value):
                valid_signals += 1

        return valid_signals

    def _empty_relational_analytics(self) -> Dict[str, Any]:
        directional = {
            "causal_energy": 0.0,
            "causal_asymmetry": 0.0,
            "causal_divergence": 0.0,
        }
        return {
            "directional": directional,
            "early_warning": {"variance": 0.0, "lag1_autocorrelation": 0.0},
            "subsystems": {
                "subsystem_count": 0.0,
                "max_subsystem_radius": 0.0,
                "subsystem_instability": 0.0,
            },
            "composite_instability": 0.0,
        }

    def process_frame(self, frame: Dict) -> Dict:
        vector = self._vector_from_frame(frame)
        n_signals = self._count_valid_signals(frame)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        baseline = self._baseline_stats()

        if baseline is None:
            result = {
                "timestamp": frame["timestamp"],
                "site_id": frame["site_id"],
                "asset_id": frame["asset_id"],
                "state": "STABLE",
                "structural_drift_score": 0.0,
                "relational_stability_score": 1.0 if n_signals >= 2 else 0.0,
                "system_health": 100,
                "drift_alert": False,
                "sensor_relationships": self.sensor_order,
                "n_signals": n_signals,
            }
            self.latest_result = result
            return result

        drift_score = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])
        stability_score = self._relational_stability() if n_signals >= 2 else 0.0
        health = self._system_health(drift_score, stability_score)
        state = self._alert_state(drift_score)
        alert = self._drift_alert(drift_score)

        result = {
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": state,
            "structural_drift_score": round(drift_score, 4),
            "relational_stability_score": round(stability_score, 4),
            "system_health": health,
            "drift_alert": alert,
            "sensor_relationships": self.sensor_order,
            "n_signals": n_signals,
        }

        if n_signals < 2:
            result["experimental_analytics"] = self._empty_relational_analytics()
            self.latest_result = result
            return result

        if len(self.frames) >= max(self.recent_window, 3):
            recent_vectors = np.vstack([f["_vector"] for f in list(self.frames)[-self.recent_window :]])
            corr = correlation_matrix(recent_vectors)
            directional = directional_metrics(lagged_correlation_matrix(recent_vectors, lag=1))
            warning = early_warning_metrics(recent_vectors)
            subsystem = subsystem_spectral_measures(corr)

            components = {
                "drift": float(drift_score),
                "spectral": spectral_radius(corr) + max(0.0, 1.0 - spectral_gap(corr)),
                "directional": directional.get("causal_divergence"),
                "entropy": interaction_entropy(corr),
                "early_warning": warning.get("variance", 0.0) + max(0.0, warning.get("lag1_autocorrelation", 0.0)),
                "subsystem_instability": subsystem.get("subsystem_instability"),
            }
            result["experimental_analytics"] = {
                "directional": directional,
                "early_warning": warning,
                "subsystems": subsystem,
                "composite_instability": round(composite_instability_score(components), 4),
            }

        self.latest_result = result
        return result
