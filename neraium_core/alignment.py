import math
from collections import deque
from typing import Dict, List, Optional

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
        self._frames = deque(maxlen=500)
        self._sensor_order: List[str] = []

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        sensor_values = frame["sensor_values"]

        if not self._sensor_order:
            self._sensor_order = sorted(sensor_values.keys())

        values = []
        for name in self._sensor_order:
            v = sensor_values.get(name, 0.0)
            try:
                values.append(float(v))
            except (TypeError, ValueError):
                values.append(0.0)

        return np.array(values, dtype=float)

    def _baseline_stats(self) -> Optional[Dict]:
        if len(self._frames) < max(5, self.baseline_window):
            return None

        baseline_frames = list(self._frames)[: self.baseline_window]
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
        if len(self._frames) < max(self.baseline_window + self.recent_window, 10):
            return 1.0

        baseline_frames = list(self._frames)[: self.baseline_window]
        recent_frames = list(self._frames)[-self.recent_window :]

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

    def process_frame(self, frame: Dict) -> Dict:
        vector = self._vector_from_frame(frame)

        stored = dict(frame)
        stored["_vector"] = vector
        self._frames.append(stored)

        baseline = self._baseline_stats()

        drift_score = 0.0
        if baseline is not None:
            drift_score = self._mahalanobis(vector, baseline["mean"], baseline["inv_cov"])

        stability_score = self._relational_stability()

        result = {
            "drift": round(float(drift_score), 4),
            "spectral": 0.0,
            "directional": 0.0,
            "entropy": 0.0,
            "early_warning": 0.0,
            "subsystem": 0.0,
            "composite_score": 0.0,
        }

        if len(self._frames) >= max(self.recent_window, 3):
            recent_vectors = np.vstack([f["_vector"] for f in list(self._frames)[-self.recent_window :]])
            corr = correlation_matrix(recent_vectors)
            directional = directional_metrics(lagged_correlation_matrix(recent_vectors, lag=1))
            warning = early_warning_metrics(recent_vectors)
            subsystem = subsystem_spectral_measures(corr)

            spectral_value = spectral_radius(corr) + max(0.0, 1.0 - spectral_gap(corr))
            directional_value = directional["causal_divergence"]
            entropy_value = interaction_entropy(corr)
            warning_value = warning["variance"] + max(0.0, warning["lag1_autocorrelation"])
            subsystem_value = subsystem["subsystem_instability"]

            components = {
                "drift": float(drift_score),
                "spectral": spectral_value,
                "directional": directional_value,
                "entropy": entropy_value,
                "early_warning": warning_value,
                "subsystem_instability": subsystem_value,
            }

            result.update(
                {
                    "spectral": round(float(spectral_value), 4),
                    "directional": round(float(directional_value), 4),
                    "entropy": round(float(entropy_value), 4),
                    "early_warning": round(float(warning_value), 4),
                    "subsystem": round(float(subsystem_value), 4),
                    "composite_score": round(composite_instability_score(components), 4),
                }
            )

        result["relational_stability"] = round(float(stability_score), 4)
        return result

    def reset(self) -> None:
        self._frames.clear()
        self._sensor_order = []
