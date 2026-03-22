from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from neraium_core.sii.confidence import confidence_to_level, estimate_confidence
from neraium_core.sii.config import SIIConfig
from neraium_core.sii.context import ExternalContextProvider, NoOpContextProvider
from neraium_core.sii.decision import map_decision_state_with_config, map_to_interpreted_state
from neraium_core.sii.errors import SIIError, SIIProcessingError
from neraium_core.sii.explanation import build_explanation_text, dominant_drivers
from neraium_core.sii.geometry_layer import build_geometry_state, flatten_upper
from neraium_core.sii.graph_layer import graph_state
from neraium_core.sii.ingestion import (
    canonical_records_from_payloads,
    frames_from_csv,
)
from neraium_core.sii.preprocessing import data_quality, impute_column_mean, summarize_quality
from neraium_core.sii.regime_model import RegimeModel, RegimeObservation
from neraium_core.sii.scoring import StructuralScoringModel
from neraium_core.sii.types import (
    ALLOWED_CONFIDENCE,
    ALLOWED_INTERPRETED_STATES,
    ALLOWED_STATES,
    CanonicalIngestionRecord,
    SIIResult,
    StructuralIndicators,
    TelemetryFrame,
    ingestion_record_to_frame,
)


@dataclass
class _State:
    vectors: deque[np.ndarray]
    timestamps: deque[float]
    sensor_order: list[str]
    baseline_corr: np.ndarray | None
    baseline_adj: np.ndarray | None
    composite_history: deque[float]
    interpreted_history: deque[str]
    raw_structural_history: deque[float]
    raw_relational_history: deque[float]
    raw_graph_history: deque[float]
    raw_regime_history: deque[float]
    raw_coherence_loss_history: deque[float]
    raw_mean_shift_history: deque[float]
    raw_cov_shift_history: deque[float]
    raw_subspace_shift_history: deque[float]
    raw_path_shift_history: deque[float]
    processed_frames: int = 0


class SystemicInfrastructureIntelligenceEngine:
    """
    Read-only Systemic Infrastructure Intelligence platform engine.

    This engine instruments multivariate structural behavior:
    - statistical geometry of relational state
    - graph topology deformation
    - regime departure from historically stable operating structure
    """

    def __init__(
        self,
        config: SIIConfig | None = None,
        *,
        context_provider: ExternalContextProvider | None = None,
    ) -> None:
        self.config = config or SIIConfig.from_env()
        self.context_provider = context_provider or NoOpContextProvider()
        self.scoring = StructuralScoringModel()
        self.regimes = RegimeModel(config=self.config)
        self.logger = logging.getLogger("neraium.sii.engine")
        self.state = _State(
            vectors=deque(maxlen=self.config.max_history),
            timestamps=deque(maxlen=self.config.max_history),
            sensor_order=[],
            baseline_corr=None,
            baseline_adj=None,
            composite_history=deque(maxlen=240),
            interpreted_history=deque(maxlen=80),
            raw_structural_history=deque(maxlen=240),
            raw_relational_history=deque(maxlen=240),
            raw_graph_history=deque(maxlen=240),
            raw_regime_history=deque(maxlen=240),
            raw_coherence_loss_history=deque(maxlen=240),
            raw_mean_shift_history=deque(maxlen=240),
            raw_cov_shift_history=deque(maxlen=240),
            raw_subspace_shift_history=deque(maxlen=240),
            raw_path_shift_history=deque(maxlen=240),
        )
        self.logger.info(
            "engine_initialized",
            extra={
                "baseline_window": self.config.baseline_window,
                "recent_window": self.config.recent_window,
                "graph_edge_threshold": self.config.effective_graph_edge_threshold,
                "regime_distance_threshold": self.config.regime_distance_threshold,
            },
        )

    @staticmethod
    def _to_float_timestamp(value: str, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

    def _clear_reference_histories(self) -> None:
        self.state.raw_structural_history.clear()
        self.state.raw_relational_history.clear()
        self.state.raw_graph_history.clear()
        self.state.raw_regime_history.clear()
        self.state.raw_coherence_loss_history.clear()
        self.state.raw_mean_shift_history.clear()
        self.state.raw_cov_shift_history.clear()
        self.state.raw_subspace_shift_history.clear()
        self.state.raw_path_shift_history.clear()
        self.state.composite_history.clear()
        self.state.interpreted_history.clear()

    def _expand_schema(self, new_sensors: list[str]) -> None:
        if not new_sensors:
            return
        updated: deque[np.ndarray] = deque(maxlen=self.config.max_history)
        pad = np.full(shape=(len(new_sensors),), fill_value=np.nan, dtype=float)
        for row in self.state.vectors:
            updated.append(np.concatenate((row, pad)).astype(float, copy=False))
        self.state.vectors = updated
        self.state.sensor_order.extend(new_sensors)
        self.state.baseline_corr = None
        self.state.baseline_adj = None
        self._clear_reference_histories()
        self.logger.warning("sensor_schema_extended", extra={"new_sensors": new_sensors})

    def _ensure_sensor_order(self, frame: TelemetryFrame) -> None:
        frame_sensors = sorted(frame.sensor_values.keys())
        if not self.state.sensor_order:
            self.state.sensor_order = frame_sensors
            return
        current = set(self.state.sensor_order)
        incoming = set(frame_sensors)
        new_sensors = sorted(incoming - current)
        if new_sensors:
            self._expand_schema(new_sensors)

    def _vectorize(self, frame: TelemetryFrame) -> np.ndarray:
        vals: list[float] = []
        for s in self.state.sensor_order:
            v = frame.sensor_values.get(s)
            if v is None:
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
        return np.asarray(vals, dtype=float)

    def _history_windows(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        n = len(self.state.vectors)
        if n < self.config.recent_window:
            return None, None
        recent = np.vstack(list(self.state.vectors)[-self.config.recent_window :])
        if n >= self.config.baseline_window + self.config.recent_window:
            baseline = np.vstack(
                list(self.state.vectors)[
                    -(self.config.baseline_window + self.config.recent_window) : -self.config.recent_window
                ]
            )
        elif n >= self.config.baseline_window:
            baseline = np.vstack(list(self.state.vectors)[: self.config.baseline_window])
        else:
            baseline = np.vstack(list(self.state.vectors)[: self.config.recent_window])
        return baseline, recent

    def _classification_stability(self) -> float:
        if len(self.state.interpreted_history) < 2:
            return 1.0
        counts = Counter(self.state.interpreted_history)
        return float(max(counts.values())) / float(len(self.state.interpreted_history))

    def _trend(self) -> float:
        vals = np.asarray(list(self.state.composite_history), dtype=float)
        if vals.size < 6:
            return 0.0
        x = np.arange(vals.size, dtype=float)
        slope, _ = np.polyfit(x, vals, 1)
        return float(slope)

    @staticmethod
    def _avg_shortest_path_length(adj: np.ndarray) -> float:
        a = np.asarray(adj, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            return 0.0
        n = int(a.shape[0])
        if n <= 1:
            return 0.0
        dist = np.full((n, n), np.inf, dtype=float)
        np.fill_diagonal(dist, 0.0)
        edge_idx = np.where(a > 0.0)
        dist[edge_idx] = 1.0
        for k in range(n):
            dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
        finite = dist[np.isfinite(dist) & (dist > 0.0)]
        if finite.size == 0:
            return 0.0
        return float(np.mean(finite))

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _score_from_reference(self, value: float, history: deque[float], fallback_scale: float) -> float:
        v = float(max(0.0, value))
        scale = max(1e-6, float(fallback_scale))
        if len(history) < 12:
            return self._clamp01(v / (v + scale))
        arr = np.asarray(history, dtype=float)
        q50 = float(np.percentile(arr, 50.0))
        q90 = float(np.percentile(arr, 90.0))
        if q90 <= q50 + 1e-6:
            return self._clamp01((v - q50) / (q50 + 1e-6))
        return self._clamp01((v - q50) / (q90 - q50))

    def _geometry_signature(self, mean_delta: np.ndarray, std_delta: np.ndarray, corr: np.ndarray, coherence: float) -> np.ndarray:
        corr_flat = flatten_upper(corr)
        return np.concatenate(
            [
                np.clip(mean_delta, -6.0, 6.0),
                np.clip(std_delta, -6.0, 6.0),
                np.clip(corr_flat, -1.0, 1.0),
                np.asarray([float(max(0.0, min(1.0, coherence)))], dtype=float),
            ]
        ).astype(float, copy=False)

    def _graph_signature(self, adjacency: np.ndarray, density: float, avg_degree: float, deformation: float) -> np.ndarray:
        adj = np.asarray(adjacency, dtype=float)
        n = int(adj.shape[0]) if adj.ndim == 2 else 0
        deg = np.sum(adj, axis=1) / max(1.0, float(n - 1)) if n > 1 else np.zeros((n,), dtype=float)
        avg_path = self._avg_shortest_path_length(adj)
        return np.concatenate(
            [
                np.asarray([float(density), float(avg_degree), float(avg_path), float(deformation)], dtype=float),
                np.asarray(deg, dtype=float),
            ]
        ).astype(float, copy=False)

    def _warmup_output(self, frame: TelemetryFrame) -> SIIResult:
        out: SIIResult = {
            "timestamp": frame.timestamp,
            "site_id": frame.site_id,
            "asset_id": frame.asset_id,
            "state": "STABLE",
            "interpreted_state": "NOMINAL_STRUCTURE",
            "confidence": "low",
            "structural_drift_score": 0.0,
            "relational_instability_score": 0.0,
            "regime_distance": 0.0,
            "coherence_score": 1.0,
            "graph_deformation_score": 0.0,
            "dominant_drivers": [],
            "confidence_reasoning": ["insufficient_history"],
            "explanation": "Warmup: awaiting multivariate relational history.",
            "read_only": True,
            "system_health": 100,
            "data_quality_summary": {
                "missingness_rate": 1.0,
                "sensor_coverage": 0.0,
                "timestamp_irregularity": 0.0,
                "gate_passed": False,
                "statuses": ["DATA_QUALITY_LIMITED"],
                "valid_signal_count": 0,
                "total_sensor_count": len(self.state.sensor_order),
            },
            "experimental_analytics": {
                "components": {},
                "raw_components": {},
                "graph_metrics": {},
                "regime": {},
                "context": None,
            },
        }
        return out

    def _update_adaptive_baseline(
        self,
        *,
        recent_corr: np.ndarray,
        recent_adj: np.ndarray,
        decision_state: str,
        interpreted_state: str,
        dq_passed: bool,
    ) -> None:
        if self.state.baseline_corr is None:
            self.state.baseline_corr = np.asarray(recent_corr, dtype=float)
        if self.state.baseline_adj is None:
            self.state.baseline_adj = np.asarray(recent_adj, dtype=float)
        if self.state.baseline_corr is None or self.state.baseline_adj is None:
            return
        if self.state.processed_frames < self.config.freeze_baseline_frames:
            return
        if not dq_passed:
            return
        if decision_state != "STABLE" or interpreted_state != "NOMINAL_STRUCTURE":
            return
        alpha = float(self.config.baseline_adaptation_alpha)
        self.state.baseline_corr = alpha * self.state.baseline_corr + (1.0 - alpha) * np.asarray(recent_corr, dtype=float)
        smoothed_adj = alpha * self.state.baseline_adj + (1.0 - alpha) * np.asarray(recent_adj, dtype=float)
        self.state.baseline_adj = (smoothed_adj >= 0.5).astype(float)
        np.fill_diagonal(self.state.baseline_adj, 0.0)

    def _update_reference_histories(
        self,
        *,
        dq_passed: bool,
        decision_state: str,
        interpreted_state: str,
        raw_structural: float,
        raw_relational: float,
        raw_graph: float,
        raw_regime: float,
        raw_coherence_loss: float,
        raw_mean_shift: float,
        raw_cov_shift: float,
        raw_subspace_shift: float,
        raw_path_shift: float,
    ) -> None:
        if not dq_passed:
            return
        if decision_state != "STABLE" or interpreted_state != "NOMINAL_STRUCTURE":
            return
        self.state.raw_structural_history.append(float(max(0.0, raw_structural)))
        self.state.raw_relational_history.append(float(max(0.0, raw_relational)))
        self.state.raw_graph_history.append(float(max(0.0, raw_graph)))
        self.state.raw_regime_history.append(float(max(0.0, raw_regime)))
        self.state.raw_coherence_loss_history.append(float(max(0.0, raw_coherence_loss)))
        self.state.raw_mean_shift_history.append(float(max(0.0, raw_mean_shift)))
        self.state.raw_cov_shift_history.append(float(max(0.0, raw_cov_shift)))
        self.state.raw_subspace_shift_history.append(float(max(0.0, raw_subspace_shift)))
        self.state.raw_path_shift_history.append(float(max(0.0, raw_path_shift)))

    def close(self) -> None:
        self.regimes.save()
        self.logger.info("engine_closed", extra={"processed_frames": self.state.processed_frames})

    def process_frame(self, frame: TelemetryFrame) -> SIIResult:
        try:
            self._ensure_sensor_order(frame)
            vec = self._vectorize(frame)
            self.state.vectors.append(vec)
            self.state.timestamps.append(self._to_float_timestamp(frame.timestamp, self.state.processed_frames))
            self.state.processed_frames += 1

            baseline, recent = self._history_windows()
            if baseline is None or recent is None:
                out = self._warmup_output(frame)
                self.state.interpreted_history.append(out["interpreted_state"])
                return out

            recent_ts = list(self.state.timestamps)[-recent.shape[0] :]
            dq = data_quality(recent, recent_timestamps=recent_ts)
            baseline_imp = impute_column_mean(baseline)
            recent_imp = impute_column_mean(recent)

            geom = build_geometry_state(
                baseline_imp,
                recent_imp,
                reference_corr=self.state.baseline_corr,
            )
            gstate = graph_state(
                geom.recent_corr,
                baseline_adj=self.state.baseline_adj,
                threshold=self.config.effective_graph_edge_threshold,
                feature_names=self.state.sensor_order,
            )

            if self.state.baseline_corr is None:
                self.state.baseline_corr = np.asarray(geom.baseline_corr, dtype=float)
            if self.state.baseline_adj is None:
                self.state.baseline_adj = np.asarray(gstate.adjacency, dtype=float)

            denom = np.abs(geom.baseline_std) + 1e-6
            mean_delta = (geom.recent_mean - geom.baseline_mean) / denom
            std_delta = (geom.recent_std - geom.baseline_std) / denom
            geometry_signature = self._geometry_signature(mean_delta, std_delta, geom.recent_corr, geom.coherence_score)
            graph_signature = self._graph_signature(
                gstate.adjacency,
                gstate.density,
                gstate.avg_degree,
                gstate.l1_deformation,
            )

            reg_assign = self.regimes.observe(
                RegimeObservation(
                    geometry_signature=np.asarray(geometry_signature, dtype=float),
                    graph_signature=np.asarray(graph_signature, dtype=float),
                    feature_names=list(self.state.sensor_order),
                )
            )

            baseline_path = (
                self._avg_shortest_path_length(self.state.baseline_adj)
                if self.state.baseline_adj is not None
                else self._avg_shortest_path_length(gstate.adjacency)
            )
            current_path = self._avg_shortest_path_length(gstate.adjacency)
            path_shift = abs(current_path - baseline_path)

            raw_structural = float(max(0.0, geom.structural_drift))
            raw_relational = float(max(0.0, geom.relational_instability))
            raw_graph = float(max(0.0, 0.70 * gstate.l1_deformation + 0.30 * path_shift))
            raw_regime = float(max(0.0, reg_assign.regime_distance))
            raw_coherence_loss = float(max(0.0, 1.0 - geom.coherence_score))
            raw_mean_shift = float(max(0.0, geom.mean_shift_norm))
            raw_cov_shift = float(max(0.0, geom.covariance_shift_norm))
            raw_subspace_shift = float(max(0.0, geom.subspace_rotation))

            structural_score = self._score_from_reference(raw_structural, self.state.raw_structural_history, fallback_scale=1.0)
            relational_score = self._score_from_reference(raw_relational, self.state.raw_relational_history, fallback_scale=0.35)
            graph_score = self._score_from_reference(raw_graph, self.state.raw_graph_history, fallback_scale=0.35)
            regime_score = self._score_from_reference(
                raw_regime,
                self.state.raw_regime_history,
                fallback_scale=max(0.25, float(self.config.regime_distance_threshold)),
            )
            coherence_loss_score = self._score_from_reference(
                raw_coherence_loss,
                self.state.raw_coherence_loss_history,
                fallback_scale=0.30,
            )
            mean_shift_score = self._score_from_reference(raw_mean_shift, self.state.raw_mean_shift_history, fallback_scale=0.45)
            cov_shift_score = self._score_from_reference(raw_cov_shift, self.state.raw_cov_shift_history, fallback_scale=0.45)
            subspace_shift_score = self._score_from_reference(
                raw_subspace_shift,
                self.state.raw_subspace_shift_history,
                fallback_scale=0.35,
            )
            path_shift_score = self._score_from_reference(path_shift, self.state.raw_path_shift_history, fallback_scale=0.35)
            coupling_score = self._clamp01(0.45 * relational_score + 0.35 * graph_score + 0.20 * coherence_loss_score)

            components = {
                "structural_drift_score": structural_score,
                "relational_instability_score": relational_score,
                "regime_distance": regime_score,
                "coherence_loss_score": coherence_loss_score,
                "graph_deformation_score": graph_score,
                "coupling_instability_score": coupling_score,
            }
            component_extensions = {
                "mean_shift_score": mean_shift_score,
                "covariance_shift_score": cov_shift_score,
                "subspace_rotation_score": subspace_shift_score,
                "path_length_shift_score": path_shift_score,
            }

            indicators = StructuralIndicators(
                structural_drift_score=float(components["structural_drift_score"]),
                relational_instability_score=float(components["relational_instability_score"]),
                regime_distance=float(components["regime_distance"]),
                coherence_loss_score=float(components["coherence_loss_score"]),
                graph_deformation_score=float(components["graph_deformation_score"]),
                coupling_instability_score=float(components["coupling_instability_score"]),
                mean_shift_score=float(component_extensions["mean_shift_score"]),
                covariance_shift_score=float(component_extensions["covariance_shift_score"]),
                subspace_rotation_score=float(component_extensions["subspace_rotation_score"]),
                path_length_shift_score=float(component_extensions["path_length_shift_score"]),
            )
            composite = float(self.scoring.composite_departure_score(indicators))
            self.state.composite_history.append(composite)

            coherence_for_decision = self._clamp01(1.0 - coherence_loss_score)
            interpreted = map_to_interpreted_state(
                structural_drift=float(structural_score),
                relational_instability=float(relational_score),
                regime_distance=float(regime_score),
                coherence_score=float(coherence_for_decision),
                graph_instability=float(graph_score),
            )
            trend = self._trend()
            classification_stability = self._classification_stability()
            decision_state = map_decision_state_with_config(
                composite_score=float(composite),
                interpreted_state=interpreted,
                trend=trend,
                stability=classification_stability,
                config=self.config,
            )
            if self.state.processed_frames < int(self.config.min_samples_for_alerts):
                decision_state = "STABLE"

            self._update_adaptive_baseline(
                recent_corr=geom.recent_corr,
                recent_adj=gstate.adjacency,
                decision_state=decision_state,
                interpreted_state=interpreted,
                dq_passed=bool(dq.gate_passed),
            )
            self._update_reference_histories(
                dq_passed=bool(dq.gate_passed),
                decision_state=decision_state,
                interpreted_state=interpreted,
                raw_structural=raw_structural,
                raw_relational=raw_relational,
                raw_graph=raw_graph,
                raw_regime=raw_regime,
                raw_coherence_loss=raw_coherence_loss,
                raw_mean_shift=raw_mean_shift,
                raw_cov_shift=raw_cov_shift,
                raw_subspace_shift=raw_subspace_shift,
                raw_path_shift=path_shift,
            )

            conf = estimate_confidence(
                data_quality={
                    "missingness_rate": dq.missingness_rate,
                    "sensor_coverage": dq.sensor_coverage,
                    "timestamp_irregularity": dq.timestamp_irregularity,
                },
                component_scores=components,
                classification_stability=classification_stability,
                regime_support=float(reg_assign.regime_support),
            )
            confidence = confidence_to_level(float(conf.score))
            if confidence not in ALLOWED_CONFIDENCE:
                confidence = "low"

            driver_scores = {**components, **component_extensions}
            driver_names = dominant_drivers(driver_scores, top_k=3)
            context = None
            if self.config.allow_context_provider:
                try:
                    snap = self.context_provider.snapshot(
                        {
                            "timestamp": frame.timestamp,
                            "site_id": frame.site_id,
                            "asset_id": frame.asset_id,
                            "sensor_values": frame.sensor_values,
                        }
                    )
                    context = None if snap is None else {"source": snap.source, "payload": snap.payload}
                except Exception:
                    self.logger.exception("context_provider_snapshot_failed")
                    context = None

            system_health = int(max(0.0, min(100.0, 100.0 - (composite * 55.0))))
            out: SIIResult = {
                "timestamp": frame.timestamp,
                "site_id": frame.site_id,
                "asset_id": frame.asset_id,
                "state": decision_state,
                "interpreted_state": interpreted,
                "confidence": confidence,
                "structural_drift_score": round(float(structural_score), 4),
                "relational_instability_score": round(float(relational_score), 4),
                "regime_distance": round(float(regime_score), 4),
                "coherence_score": round(float(geom.coherence_score), 4),
                "graph_deformation_score": round(float(graph_score), 4),
                "dominant_drivers": driver_names,
                "confidence_reasoning": list(conf.reasoning),
                "explanation": build_explanation_text(
                    interpreted_state=interpreted,
                    decision_state=decision_state,
                    dominant=driver_names,
                    confidence_reasoning=list(conf.reasoning),
                ),
                "read_only": True,
                "system_health": system_health,
                "data_quality_summary": summarize_quality(dq),
                "experimental_analytics": {
                    "components": {k: round(float(v), 6) for k, v in components.items()},
                    "component_extensions": {k: round(float(v), 6) for k, v in component_extensions.items()},
                    "raw_components": {
                        "structural_drift": round(raw_structural, 6),
                        "relational_instability": round(raw_relational, 6),
                        "graph_deformation": round(raw_graph, 6),
                        "regime_distance": round(raw_regime, 6),
                        "coherence_loss": round(raw_coherence_loss, 6),
                        "mean_shift": round(raw_mean_shift, 6),
                        "covariance_shift": round(raw_cov_shift, 6),
                        "subspace_rotation": round(raw_subspace_shift, 6),
                        "path_length_shift": round(path_shift, 6),
                    },
                    "composite_instability": round(composite, 6),
                    "geometry": {
                        "mean_shift_norm": round(float(geom.mean_shift_norm), 6),
                        "covariance_shift_norm": round(float(geom.covariance_shift_norm), 6),
                        "subspace_rotation": round(float(geom.subspace_rotation), 6),
                        "coherence_score": round(float(geom.coherence_score), 6),
                    },
                    "graph_metrics": {
                        "density": round(float(gstate.density), 6),
                        "avg_degree": round(float(gstate.avg_degree), 6),
                        "path_length": round(float(current_path), 6),
                        "path_length_shift": round(float(path_shift), 6),
                        "deformation": round(float(gstate.l1_deformation), 6),
                    },
                    "regime": {
                        "name": reg_assign.regime_name,
                        "distance": reg_assign.regime_distance,
                        "geometry_distance": reg_assign.geometry_distance,
                        "graph_distance": reg_assign.graph_distance,
                        "pending": not bool(reg_assign.regime_activated),
                        "support": reg_assign.regime_support,
                    },
                    "context": context,
                },
            }
            if out["state"] not in ALLOWED_STATES:
                out["state"] = "STABLE"
            if out["interpreted_state"] not in ALLOWED_INTERPRETED_STATES:
                out["interpreted_state"] = "NOMINAL_STRUCTURE"
            self.state.interpreted_history.append(out["interpreted_state"])
            if self.state.processed_frames % 25 == 0:
                self.regimes.save()
            return out
        except SIIError:
            raise
        except Exception as exc:
            self.logger.exception("frame_processing_failure")
            raise SIIProcessingError(
                f"Failed to process telemetry frame safely for asset={frame.asset_id!r} "
                f"site={frame.site_id!r} timestamp={frame.timestamp!r}"
            ) from exc

    def process_payload(self, payload: dict[str, Any]) -> SIIResult:
        record = canonical_records_from_payloads(
            [payload],
            source_type="payload",
            source_name="engine_process_payload",
        )[0]
        return self.process_record(record)

    def process_record(self, record: CanonicalIngestionRecord) -> SIIResult:
        return self.process_frame(ingestion_record_to_frame(record))

    def process_records(self, records: list[CanonicalIngestionRecord]) -> list[SIIResult]:
        return [self.process_record(record) for record in records]

    def process_csv_text(self, csv_text: str) -> list[SIIResult]:
        frames = frames_from_csv(csv_text)
        return [self.process_frame(f) for f in frames]


SIIEngine = SystemicInfrastructureIntelligenceEngine
