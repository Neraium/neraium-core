from __future__ import annotations

from collections import Counter, deque
from typing import Dict, List, Optional

import numpy as np

from neraium_core.causal import causal_metrics, granger_causality_matrix
from neraium_core.causal_attribution import causal_attribution
from neraium_core.causal_graph import causal_graph_metrics
from neraium_core.data_quality import (
    compute_data_quality,
    data_quality_summary,
    impute_missing_simple,
    should_use_degraded_analytics,
)
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
from neraium_core.scoring import canonicalize_components, canonicalize_weights, composite_instability_score_normalized
from neraium_core.spectral import dominant_mode_loading, spectral_gap, spectral_radius
from neraium_core.staged_pipeline import (
    AttributionStage,
    DecisionStage,
    FeatureExtractionStage,
    NodeBaselineProfile,
    RelationalInstabilityStage,
    StructuralDriftStage,
    TemporalCoherenceStage,
    flatten_upper_tri,
)
from neraium_core.subsystems import subsystem_spectral_measures


# How slowly the rolling baseline adapts (only when nominal); avoid absorbing instability.
DEFAULT_BASELINE_ADAPTATION_ALPHA = 0.92
# Composite below this and nominal state required to update rolling baseline.
BASELINE_UPDATE_MAX_COMPOSITE = 0.85
# Number of recent interpreted states to compute classification stability.
CLASSIFICATION_STABILITY_WINDOW = 15


class StructuralEngine:
    def __init__(
        self,
        baseline_window: int = 50,
        recent_window: int = 12,
        window_stride: int = 1,
        regime_store_path: str = "regime_library.json",
        baseline_adaptation_alpha: float = DEFAULT_BASELINE_ADAPTATION_ALPHA,
    ):
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self.window_stride = max(1, window_stride)
        self.frames = deque(maxlen=500)
        self.sensor_order: List[str] = []
        self.latest_result: Optional[Dict] = None
        self.score_history: deque[float] = deque(maxlen=120)
        self.baseline_adaptation_alpha = baseline_adaptation_alpha
        # Rolling baseline: updated only when system is nominal and composite low.
        self._rolling_baseline_corr: Optional[np.ndarray] = None
        # Recent interpreted states for classification stability.
        self._state_history: deque[str] = deque(maxlen=CLASSIFICATION_STABILITY_WINDOW)
        self._stage_baseline_profile = NodeBaselineProfile()

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

    def _persistence_features(self) -> dict[str, float]:
        """
        Lightweight persistence/hysteresis helpers derived from composite history.

        This does not change analytics; it provides decision-layer context so
        transient motion does not escalate into persistent instability.
        """
        values = [float(v) for v in self.score_history]
        if not values:
            return {
                "history_len": 0.0,
                "rolling_mean": 0.0,
                "rolling_std": 0.0,
                "consecutive_elevated": 0.0,
                "consecutive_high": 0.0,
            }

        window = values[-min(len(values), 12) :]
        rolling_mean = float(np.mean(window)) if window else 0.0
        rolling_std = float(np.std(window)) if window else 0.0

        consecutive_elevated = 0
        consecutive_high = 0
        for v in reversed(values):
            if v >= 1.5:
                consecutive_elevated += 1
            else:
                break
        for v in reversed(values):
            if v >= 2.5:
                consecutive_high += 1
            else:
                break

        return {
            "history_len": float(len(values)),
            "rolling_mean": float(rolling_mean),
            "rolling_std": float(rolling_std),
            "consecutive_elevated": float(consecutive_elevated),
            "consecutive_high": float(consecutive_high),
        }

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

    def _get_recent_timestamps(self) -> Optional[list[float]]:
        if len(self.frames) < self.recent_window:
            return None
        ts_vals: list[float] = []
        for f in list(self.frames)[-self.recent_window:]:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

    def _get_baseline_timestamps(self) -> Optional[list[float]]:
        if len(self.frames) < self.baseline_window:
            return None
        ts_vals: list[float] = []
        for f in list(self.frames)[: self.baseline_window]:
            try:
                ts_vals.append(float(f.get("timestamp")))
            except (TypeError, ValueError):
                continue
        return ts_vals if len(ts_vals) >= 2 else None

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
            "relational_instability_score": 0.0,
            "temporal_distortion_score": 0.0,
            "localization_score": 0.0,
            "causal_attribution": {"top_drivers": [], "driver_scores": {}},
            "dominant_driver": None,
            "explanation": "Warmup: awaiting sufficient window history.",
            "baseline_mode": None,
            "data_quality_summary": {},
            "active_sensor_count": 0,
            "missing_sensor_count": 0,
        }

        baseline_window = self._get_baseline_window()
        recent_window = self._get_recent_window()

        if baseline_window is None or recent_window is None:
            self.latest_result = result
            return result

        data_quality_report = compute_data_quality(
            baseline_window,
            recent_window,
            sensor_names=self.sensor_order,
            timestamps_baseline=self._get_baseline_timestamps(),
            timestamps_recent=self._get_recent_timestamps(),
        )
        result["data_quality"] = data_quality_report.to_dict()
        dq_summary = data_quality_summary(data_quality_report)
        result["data_quality_summary"] = dq_summary
        result["active_sensor_count"] = dq_summary["valid_signal_count"]
        result["missing_sensor_count"] = dq_summary["missing_sensor_count"]

        use_degraded = (not data_quality_report.gate_passed) and should_use_degraded_analytics(
            data_quality_report
        )
        # Optional imputation when gate failed but we still want meaningful degraded output.
        if not data_quality_report.gate_passed and use_degraded:
            baseline_window = impute_missing_simple(baseline_window, method="column_mean")
            recent_window = impute_missing_simple(recent_window, method="column_mean")

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
            stage_features = FeatureExtractionStage.extract(z_base_valid, z_recent_valid)

            corr_baseline = correlation_matrix(z_base_valid)
            corr_recent = correlation_matrix(z_recent_valid)

            # Adaptive baseline: use rolling baseline when available to avoid static reference.
            baseline_corr_used = corr_baseline
            baseline_mode = "fixed"
            if (
                self._rolling_baseline_corr is not None
                and self._rolling_baseline_corr.shape == corr_recent.shape
            ):
                baseline_corr_used = self._rolling_baseline_corr
                baseline_mode = "rolling"

            self._stage_baseline_profile.corr_baseline = np.array(baseline_corr_used, dtype=float, copy=True)
            stage_structural_raw, _ = StructuralDriftStage.score(stage_features, self._stage_baseline_profile)
            stage_relational_raw, _ = RelationalInstabilityStage.score(stage_features, self._stage_baseline_profile)
            temporal_raw, _ = TemporalCoherenceStage.score(self._get_recent_timestamps(), self._stage_baseline_profile)
            # Preserve production sensitivity by keeping legacy drift geometry while
            # binding stage outputs into runtime diagnostics.
            drift_score = structural_drift(corr_recent, baseline_corr_used, norm="fro")
            rel_delta_legacy = flatten_upper_tri(corr_recent) - flatten_upper_tri(baseline_corr_used)
            relational_raw = float(np.mean(np.abs(rel_delta_legacy))) if rel_delta_legacy.size else 0.0
            relational_raw = max(relational_raw, stage_relational_raw, 0.5 * stage_structural_raw)
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
                    # Regime-specific baseline: EMA update so we gradually adapt inside stable regime.
                    alpha = 0.88
                    updated = alpha * regime_corr + (1.0 - alpha) * corr_recent
                    self.regime_baselines[regime_name]["correlation"] = updated.tolist()
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

            valid_sensor_names = [self.sensor_order[i] for i in range(len(valid_mask)) if valid_mask[i]]
            attr = causal_attribution(
                baseline_corr_used,
                corr_recent,
                causal_matrix,
                valid_sensor_names,
                top_k=10,
            )
            result["causal_attribution"] = attr
            result["dominant_driver"] = attr["top_drivers"][0] if attr["top_drivers"] else None

            subsystem = subsystem_spectral_measures(corr_recent)

            spectral = {
                "radius": spectral_radius(corr_recent),
                "gap": spectral_gap(corr_recent),
                **dominant_mode_loading(corr_recent),
            }

            raw_components = {
                "drift": drift_score,
                "relational_drift": relational_raw,
                "regime_drift": regime_drift,
                "spectral": spectral["radius"],
                "directional": max(
                    float(directional.get("divergence", 0.0)),
                    float(causal.get("causal_divergence", 0.0)),
                ),
                "entropy": interaction_entropy(corr_recent),
                "subsystem_instability": float(subsystem["max_instability"]),
                "temporal_distortion": temporal_raw,
            }

            # Merge order matters: preserve early_warning computed from the
            # latest signal window, while ensuring freshly computed relational
            # drift / regime drift / spectral / divergence / entropy /
            # subsystem instability are not clobbered by stale base defaults.
            base_components = components
            raw_canonical = canonicalize_components(raw_components)
            raw_canonical["early_warning"] = float(base_components.get("early_warning", 0.0))

            base_components.update(raw_canonical)
            components = base_components

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
                    "baseline_mode": baseline_mode,
                }
            )
            regime_memory_state = {
                "regime_name": regime_name,
                "library_size": len(self.regime_signatures),
                "baseline_count": (
                    int(self.regime_baselines.get(regime_name, {}).get("count", 0))
                    if regime_name
                    else None
                ),
            }
            result["regime_memory_state"] = regime_memory_state

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
        else:
            result["regime_memory_state"] = {
                "regime_name": regime_name,
                "library_size": len(self.regime_signatures),
                "baseline_count": None,
            }

        # Per-component confidence: down-weight or fully suppress evidence when the
        # data quality gate indicates unreliable inputs. Production alerts should
        # be driven by Tier-1 components only.
        tier1_components = {"relational_drift", "regime_drift", "spectral", "early_warning"}

        # Evidence quality in [0, 1]
        missingness_factor = max(0.0, 1.0 - float(data_quality_report.missingness_rate))
        variability_factor = max(0.0, min(1.0, float(data_quality_report.variability_coverage)))
        coverage_factor = max(0.0, min(1.0, float(data_quality_report.sensor_coverage)))
        sample_factor = 0.0
        if data_quality_report.total_sensors > 0:
            sample_factor = float(data_quality_report.valid_signal_count) / float(max(1, data_quality_report.total_sensors))
        sample_factor = max(0.0, min(1.0, sample_factor))

        evidence_conf = (
            missingness_factor
            * (0.4 + 0.6 * variability_factor)
            * (0.4 + 0.6 * coverage_factor)
            * (0.5 + 0.5 * sample_factor)
        )
        if not bool(data_quality_report.gate_passed):
            evidence_conf *= 0.25
        if use_degraded:
            evidence_conf *= 0.5  # Explicit degraded confidence when using fallback analytics
        evidence_conf = max(0.0, min(1.0, evidence_conf))

        correlation_ready = valid_signal_count >= 2

        # Classification stability: how consistent recent interpreted states have been.
        state_history_list = list(self._state_history)
        if len(state_history_list) >= 2:
            counts = Counter(state_history_list)
            most_common_count = max(counts.values()) if counts else 0
            classification_stability = float(most_common_count) / float(len(state_history_list))
        else:
            classification_stability = 1.0

        # Metric disagreement: high std across components slightly reduces confidence.
        comp_vals = [float(components.get(k, 0.0)) for k in tier1_components if k in components]
        if comp_vals:
            mean_c = sum(comp_vals) / len(comp_vals)
            std_c = (sum((x - mean_c) ** 2 for x in comp_vals) / len(comp_vals)) ** 0.5
            disagreement = std_c / (mean_c + 1e-6)
            disagreement_factor = max(0.7, 1.0 - disagreement * 0.15)
        else:
            disagreement_factor = 1.0

        stabilized_confidence = evidence_conf * (0.6 + 0.4 * classification_stability) * disagreement_factor
        stabilized_confidence = max(0.0, min(1.0, stabilized_confidence))

        # Regime baseline confidence depends on how much history exists for the
        # assigned regime. If we don't yet have baseline correlation samples,
        # the regime drift evidence is treated as unreliable.
        regime_count = 0
        if regime_name is not None:
            entry = self.regime_baselines.get(regime_name)
            if isinstance(entry, dict):
                try:
                    regime_count = int(entry.get("count", 0) or 0)
                except (TypeError, ValueError):
                    regime_count = 0

        regime_factor = min(1.0, float(regime_count) / 5.0) if regime_count > 0 else 0.0

        component_confidence: dict[str, float] = {k: 0.0 for k in components.keys()}

        # Tier-1
        component_confidence["relational_drift"] = evidence_conf if correlation_ready else 0.0
        component_confidence["spectral"] = evidence_conf if correlation_ready else 0.0
        component_confidence["early_warning"] = evidence_conf
        component_confidence["regime_drift"] = evidence_conf * regime_factor if correlation_ready else 0.0

        # Suppress non-Tier-1 components explicitly (keeps production composite Tier-1 only)
        for k in list(component_confidence.keys()):
            if k not in tier1_components:
                component_confidence[k] = 0.0

        analytics["component_confidence"] = component_confidence

        # Confidence-weighted composite: use confidence as a scaling on component weights
        # so that unreliable evidence doesn't dilute the Tier-1 score.
        base_weights = canonicalize_weights()
        weights_for_composite: dict[str, float] = {}
        for k, w in base_weights.items():
            weights_for_composite[k] = float(w) * float(component_confidence.get(k, 0.0))

        components_for_decision = {
            k: float(v) * float(component_confidence.get(k, 0.0)) if k in component_confidence else float(v)
            for k, v in components.items()
        }

        composite = composite_instability_score_normalized(components, weights=weights_for_composite)
        self.score_history.append(float(composite))

        persistence = self._persistence_features()

        forecast = {
            "method": "regression+ar1",
            "trend": float(instability_trend(self.score_history)),
            "time_to_instability": time_to_instability(self.score_history),
            "ar1_next": forecast_next(self.score_history),
            "ar1_time_to_instability": time_to_threshold_ar1(self.score_history),
            "persistence": persistence,
        }

        decision = decision_output(
            composite_score=float(composite),
            components=components_for_decision,
            forecast=forecast,
            confidence_score=stabilized_confidence,
            classification_stability=classification_stability,
        )
        result.update(decision)
        stage_interpreted = DecisionStage.interpreted_state(
            structural=float(components.get("drift", 0.0)),
            relational=float(components.get("relational_drift", 0.0)),
            regime_distance=float(components.get("regime_drift", 0.0)),
            temporal_distortion=float(components.get("temporal_distortion", 0.0)),
            localization=1.0,
            trend=float(forecast.get("trend", 0.0)),
        )
        if (
            str(result.get("interpreted_state", "NOMINAL_STRUCTURE")) == "NOMINAL_STRUCTURE"
            and stage_interpreted != "NOMINAL_STRUCTURE"
        ):
            result["interpreted_state"] = stage_interpreted
        elif str(result.get("interpreted_state", "NOMINAL_STRUCTURE")) == "NOMINAL_STRUCTURE":
            # Single-node runtime fallback: preserve legacy structural/coupling detection
            # semantics when multi-node localization context is unavailable.
            rel = float(components.get("relational_drift", 0.0))
            drf = float(components.get("drift", 0.0))
            if rel > 0.9:
                result["interpreted_state"] = "COUPLING_INSTABILITY_OBSERVED"
            elif drf > 1.1:
                result["interpreted_state"] = "STRUCTURAL_INSTABILITY_OBSERVED"
        result["confidence_score"] = round(stabilized_confidence, 4)
        result["latest_instability"] = round(float(composite), 4)
        result["relational_instability_score"] = round(float(components.get("relational_drift", 0.0)), 4)
        result["temporal_distortion_score"] = round(float(components.get("temporal_distortion", data_quality_report.timestamp_irregularity)), 4)
        result["localization_score"] = 0.0

        self._state_history.append(decision.get("interpreted_state", "NOMINAL_STRUCTURE"))

        # Rolling baseline: update only when nominal and composite low (avoid absorbing instability).
        if (
            valid_signal_count >= 2
            and decision.get("interpreted_state") == "NOMINAL_STRUCTURE"
            and float(composite) < BASELINE_UPDATE_MAX_COMPOSITE
        ):
            if self._rolling_baseline_corr is None or self._rolling_baseline_corr.shape != corr_recent.shape:
                self._rolling_baseline_corr = np.array(corr_recent, dtype=float, copy=True)
            else:
                alpha = self.baseline_adaptation_alpha
                self._rolling_baseline_corr = alpha * self._rolling_baseline_corr + (1.0 - alpha) * corr_recent

        analytics["composite_instability"] = round(float(composite), 4)
        analytics["forecasting"] = forecast
        analytics["components"] = components
        explain_components = {
            "structural_drift_score": float(result.get("structural_drift_score", 0.0)),
            "relational_instability_score": float(result.get("relational_instability_score", 0.0)),
            "regime_distance": float(result.get("regime_distance", 0.0) or 0.0),
            "temporal_distortion_score": float(result.get("temporal_distortion_score", 0.0)),
        }
        msg, contrib = AttributionStage.explain(explain_components, str(result.get("state", "STABLE")))
        result["explanation"] = msg
        analytics["component_contributions"] = contrib
        result["dominant_driver"] = (
            max(contrib.items(), key=lambda item: item[1])[0]
            if contrib
            else result.get("dominant_driver")
        )
        result["component_confidence"] = component_confidence

        result["experimental_analytics"] = analytics
        self.latest_result = result

        return result