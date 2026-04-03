from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from neraium_core.directional import directional_metrics, lagged_correlation_matrix
from neraium_core.early_warning import early_warning_metrics
from neraium_core.entropy import interaction_entropy
from neraium_core.forecasting import instability_trend, time_to_instability
from neraium_core.geometry import (
    correlation_matrix,
    normalize_window,
    signal_structural_importance,
    structural_drift,
)
from neraium_core.graph import graph_metrics, thresholded_adjacency
from neraium_core.scoring import canonicalize_components, composite_instability_score
from neraium_core.spectral import dominant_mode_loading, spectral_gap, spectral_radius
from neraium_core.subsystems import subsystem_spectral_measures


class StructuralEngine:
    def __init__(self, baseline_window: int = 50, recent_window: int = 12, window_stride: int = 1):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.window_stride = max(1, window_stride)
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.score_history: deque[float] = deque(maxlen=120)
        self.regime_signatures: list[dict[str, object]] = []

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        sensor_values = frame["sensor_values"]

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values = []
        for name in self.sensor_order:
            v = sensor_values.get(name)
            try:
                if v is None:
                    values.append(np.nan)
                else:
                    values.append(float(v))
            except (TypeError, ValueError):
                values.append(np.nan)

        return np.array(values, dtype=float)

    def _get_recent_window(self) -> Optional[np.ndarray]:
        if len(self.frames) < self.recent_window:
            return None
        vectors = np.vstack([f["_vector"] for f in list(self.frames)[-self.recent_window :]])
        return vectors[:: self.window_stride]

    def _get_baseline_window(self) -> Optional[np.ndarray]:
        if len(self.frames) < self.baseline_window:
            return None
        vectors = np.vstack([f["_vector"] for f in list(self.frames)[: self.baseline_window]])
        return vectors[:: self.window_stride]

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        health = 100.0 - min(drift_score * 20.0, 85.0)
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

    def _nearest_regime(self, signature: np.ndarray) -> dict[str, float] | None:
        if not self.regime_signatures:
            return None
        distances = []
        for regime in self.regime_signatures:
            centroid = np.asarray(regime["signature"], dtype=float)
            distances.append((float(np.linalg.norm(signature - centroid)), str(regime["name"])))
        distances.sort(key=lambda x: x[0])
        return {"name": distances[0][1], "distance": distances[0][0]}

    def process_frame(self, frame: Dict) -> Dict:
        vector = self._vector_from_frame(frame)

        stored = dict(frame)
        stored["_vector"] = vector
        self.frames.append(stored)

        result = {
            "timestamp": frame["timestamp"],
            "site_id": frame["site_id"],
            "asset_id": frame["asset_id"],
            "state": "STABLE",
            "structural_drift_score": 0.0,
            "relational_stability_score": 1.0,
            "system_health": 100,
            "drift_alert": False,
            "sensor_relationships": self.sensor_order,
        }

        baseline_window = self._get_baseline_window()
        recent_window = self._get_recent_window()

        if baseline_window is None or recent_window is None:
            self.latest_result = result
            return result

        z_baseline, baseline_mean, baseline_std = normalize_window(baseline_window)
        z_recent, recent_mean, recent_std = normalize_window(recent_window)

        valid_mask = (np.nan_to_num(recent_std) > 1e-12) | (np.nan_to_num(baseline_std) > 1e-12)
        valid_signal_count = int(np.sum(valid_mask))

        warning = early_warning_metrics(np.nan_to_num(recent_window, nan=0.0))
        analytics: dict[str, object] = {
            "normalization": {
                "window": "zscore",
                "means": [float(v) for v in recent_mean],
                "std": [float(v) for v in recent_std],
            },
            "windowing": {
                "baseline_window": self.baseline_window,
                "recent_window": self.recent_window,
                "stride": self.window_stride,
            },
            "early_warning": warning,
            "relational_metrics_skipped": valid_signal_count < 2,
            "regime_signature": {
                "current": [float(v) for v in np.concatenate([recent_mean, recent_std])],
                "nearest": None,
            },
        }

        components = canonicalize_components(
            {
                "drift": 0.0,
                "early_warning": warning["variance"] + max(0.0, warning["lag1_autocorrelation"]),
            }
        )

        if valid_signal_count >= 2:
            z_base_valid = z_baseline[:, valid_mask]
            z_recent_valid = z_recent[:, valid_mask]
            corr_baseline = correlation_matrix(z_base_valid)
            corr_recent = correlation_matrix(z_recent_valid)
            drift_score = structural_drift(corr_recent, corr_baseline, norm="fro")
            stability_score = 1.0 / (1.0 + drift_score)

            signal_importance = signal_structural_importance(corr_recent)
            adjacency = thresholded_adjacency(corr_recent, threshold=0.6)
            graph = graph_metrics(adjacency, corr=corr_recent)
            directional = directional_metrics(lagged_correlation_matrix(z_recent_valid, lag=1))
            subsystem = subsystem_spectral_measures(corr_recent)
            spectral = {
                "radius": spectral_radius(corr_recent),
                "gap": spectral_gap(corr_recent),
                **dominant_mode_loading(corr_recent),
            }

            components.update(
                canonicalize_components(
                    {
                        "drift": drift_score,
                        "spectral": spectral["radius"] + max(0.0, 1.0 - spectral["gap"]),
                        "directional": directional["causal_divergence"],
                        "entropy": interaction_entropy(corr_recent),
                        "subsystem_instability": subsystem["subsystem_instability"],
                    }
                )
            )
            result.update(
                {
                    "structural_drift_score": round(drift_score, 4),
                    "relational_stability_score": round(stability_score, 4),
                    "system_health": self._system_health(drift_score, stability_score),
                    "state": self._alert_state(drift_score),
                    "drift_alert": self._drift_alert(drift_score),
                }
            )

            analytics.update(
                {
                    "correlation_geometry": {
                        "baseline": corr_baseline.tolist(),
                        "current": corr_recent.tolist(),
                    },
                    "signal_structural_importance": [float(v) for v in signal_importance],
                    "graph": graph,
                    "directional": directional,
                    "subsystems": subsystem,
                    "spectral": spectral,
                    "entropy": float(interaction_entropy(corr_recent)),
                }
            )

        composite = composite_instability_score(components)
        self.score_history.append(composite)
        forecast = {
            "heuristic": True,
            "trend": float(instability_trend(self.score_history)),
            "time_to_instability": time_to_instability(self.score_history, threshold=1.5),
        }

        signature = np.asarray(analytics["regime_signature"]["current"], dtype=float)
        nearest = self._nearest_regime(signature)
        analytics["regime_signature"]["nearest"] = nearest
        if nearest is None:
            self.regime_signatures.append({"name": "bootstrap_regime", "signature": signature.tolist()})

        analytics["composite_instability"] = round(float(composite), 4)
        analytics["composite_components"] = components
        analytics["forecasting"] = forecast

        result["experimental_analytics"] = analytics
        self.latest_result = result
        return result
