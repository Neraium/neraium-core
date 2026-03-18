from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from neraium_core.causal import causal_metrics, granger_causality_matrix
from neraium_core.causal_graph import causal_graph_metrics
from neraium_core.decision_layer import decision_output
from neraium_core.directional import directional_metrics, lagged_correlation_matrix
from neraium_core.early_warning import early_warning_metrics
from neraium_core.entropy import interaction_entropy
from neraium_core.forecast_models import forecast_next, time_to_threshold_ar1
from neraium_core.forecasting import instability_trend, time_to_instability
from neraium_core.geometry import (
    correlation_matrix,
    normalize_window,
    signal_structural_importance,
    structural_drift,
)
from neraium_core.graph import graph_metrics, thresholded_adjacency
from neraium_core.regime import build_regime_signature, assign_regime, update_regime_library
from neraium_core.regime_store import RegimeStore
from neraium_core.scoring import canonicalize_components, composite_instability_score
from neraium_core.spectral import dominant_mode_loading, spectral_gap, spectral_radius
from neraium_core.subsystems import subsystem_spectral_measures


class StructuralEngine:
    def __init__(
        self,
        baseline_window: int = 50,
        recent_window: int = 12,
        window_stride: int = 1,
        regime_store_path: str = "regime_library.json",
    ):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.window_stride = max(1, window_stride)
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.score_history: deque[float] = deque(maxlen=120)

        self.regime_store = RegimeStore(regime_store_path)
        persisted = self.regime_store.load()
        self.regime_signatures: list[dict[str, object]] = list(persisted.get("regimes", []))
        self.regime_baselines: dict[str, dict[str, object]] = dict(persisted.get("baselines", {}))

    def _persist_regime_state(self) -> None:
        self.regime_store.save(
            {
                "regimes": self.regime_signatures,
                "baselines": self.regime_baselines,
            }
        )

    def _vector_from_frame(self, frame: Dict) -> np.ndarray:
        sensor_values = frame["sensor_values"]

        if not self.sensor_order:
            self.sensor_order = sorted(sensor_values.keys())

        values = []
        for name in self.sensor_order:
            v = sensor_values.get(name)
            try:
                values.append(float(v) if v is not None else np.nan)
            except (TypeError, ValueError):
                values.append(np.nan)

        return np.array(values, dtype=float)

    def _get_recent_window(self) -> Optional[np.ndarray]:
        if len(self.frames) < self.recent_window:
            return None

        vectors = np.vstack([f["_vector"] for f in list(self.frames)[-self.recent_window:]])
        vectors = vectors[:: self.window_stride]

        if vectors.shape[0] < 2:
            return None

        return vectors

    def _get_baseline_window(self) -> Optional[np.ndarray]:
        if len(self.frames) < self.baseline_window:
            return None

        vectors = np.vstack([f["_vector"] for f in list(self.frames)[: self.baseline_window]])
        vectors = vectors[:: self.window_stride]

        if vectors.shape[0] < 2:
            return None

        return vectors

    def _system_health(self, drift_score: float, stability_score: float) -> int:
        health = 100.0 - min(drift_score * 20.0, 85.0)
        health += stability_score * 20.0
        return int(round(max(0.0, min(100.0, health))))

    def _alert_state(self, drift_score: float) -> str:
        if drift_score > 3.0:
            return "ALERT"
        if drift_score > 1.5:
            return "WATCH"
        return "STABLE"

    def _drift_alert(self, drift_score: float) -> bool:
        return drift_score > 1.5

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
            "regime_name": None,
            "regime_distance": None,
            "regime_drift": 0.0,
            "latest_drift": 0.0,
            "latest_instability": 0.0,
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

        signature = build_regime_signature(recent_mean, recent_std)
        assigned_regime = assign_regime(signature, self.regime_signatures)
        self.regime_signatures = update_regime_library(signature, self.regime_signatures)
        assigned_regime = assign_regime(signature, self.regime_signatures)

        regime_name = assigned_regime["name"] if assigned_regime else None
        regime_distance = float(assigned_regime["distance"]) if assigned_regime else None

        analytics: dict[str, object] = {
            "early_warning": warning,
            "relational_metrics_skipped": valid_signal_count < 2,
            "regime_signature": {
                "current": [float(v) for v in signature],
                "nearest": assigned_regime,
                "assigned_name": regime_name,
                "library_size": len(self.regime_signatures),
            },
        }

        components = canonicalize_components(
            {
                "drift": 0.0,
                "regime_drift": 0.0,
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

            regime_drift = 0.0
            if regime_name is not None:
                if regime_name not in self.regime_baselines:
                    self.regime_baselines[regime_name] = {
                        "signature": signature.tolist(),
                        "correlation": corr_recent.tolist(),
                        "count": 1,
                    }
                else:
                    regime_corr = np.asarray(self.regime_baselines[regime_name]["correlation"], dtype=float)
                    regime_drift = structural_drift(corr_recent, regime_corr, norm="fro")
                    self.regime_baselines[regime_name]["count"] = int(
                        self.regime_baselines[regime_name].get("count", 0)
                    ) + 1

                self._persist_regime_state()

            signal_importance = signal_structural_importance(corr_recent)
            adjacency = thresholded_adjacency(corr_recent, threshold=0.6)
            graph = graph_metrics(adjacency, corr=corr_recent)

            directional = directional_metrics(lagged_correlation_matrix(z_recent_valid, lag=1))

            causal_matrix = granger_causality_matrix(z_recent_valid)
            causal = causal_metrics(causal_matrix)
            causal_graph = causal_graph_metrics(causal_matrix, threshold=0.1)

            subsystem = subsystem_spectral_measures(corr_recent)

            spectral = {
                "radius": spectral_radius(corr_recent),
                "gap": spectral_gap(corr_recent),
                **dominant_mode_loading(corr_recent),
            }

            raw_components = {
                "drift": drift_score,
                "regime_drift": regime_drift,
                "spectral": spectral["radius"],
                "directional": max(
                    float(directional.get("divergence", 0.0)),
                    float(causal.get("causal_divergence", 0.0)),
                ),
                "entropy": interaction_entropy(corr_recent),
                "subsystem_instability": float(subsystem["max_instability"]),
            }

            canonical = canonicalize_components(raw_components)
            canonical.update(components)
            components = canonical

            result.update(
                {
                    "structural_drift_score": round(drift_score, 4),
                    "relational_stability_score": round(stability_score, 4),
                    "system_health": self._system_health(drift_score, stability_score),
                    "state": self._alert_state(drift_score),
                    "drift_alert": self._drift_alert(drift_score),
                    "regime_name": regime_name,
                    "regime_distance": round(regime_distance, 4) if regime_distance is not None else None,
                    "regime_drift": round(float(regime_drift), 4),
                    "latest_drift": round(float(drift_score), 4),
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
                    "causal": causal,
                    "causal_graph": causal_graph,
                    "subsystems": subsystem,
                    "spectral": spectral,
                    "entropy": float(interaction_entropy(corr_recent)),
                    "regime_drift": float(regime_drift),
                }
            )

        composite = composite_instability_score(components)
        self.score_history.append(float(composite))

        forecast = {
            "method": "regression+ar1",
            "trend": float(instability_trend(self.score_history)),
            "time_to_instability": time_to_instability(self.score_history),
            "ar1_next": forecast_next(self.score_history),
            "ar1_time_to_instability": time_to_threshold_ar1(self.score_history),
        }

        decision = decision_output(
            composite_score=float(composite),
            components=components,
            forecast=forecast,
        )
        result.update(decision)
        result["latest_instability"] = round(float(composite), 4)

        analytics["composite_instability"] = round(float(composite), 4)
        analytics["forecasting"] = forecast
        analytics["components"] = components

        result["experimental_analytics"] = analytics
        self.latest_result = result

        return result