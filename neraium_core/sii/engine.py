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
from neraium_core.sii.geometry_layer import build_geometry_state
from neraium_core.sii.graph_layer import graph_state
from neraium_core.sii.ingestion import frame_from_payload, frames_from_csv
from neraium_core.sii.preprocessing import data_quality, impute_column_mean, summarize_quality
from neraium_core.sii.regime_model import RegimeModel, RegimeObservation
from neraium_core.sii.scoring import StructuralScoringModel
from neraium_core.sii.types import (
    ALLOWED_CONFIDENCE,
    ALLOWED_INTERPRETED_STATES,
    ALLOWED_STATES,
    SIIResult,
    StructuralIndicators,
    TelemetryFrame,
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
    processed_frames: int = 0


class SystemicInfrastructureIntelligenceEngine:
    """
    Read-only Systemic Infrastructure Intelligence platform engine.

    This engine instruments multivariate structural behavior:
    - statistical geometry of relational state
    - graph topology deformation
    - regime departure from known stable configurations
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
            composite_history=deque(maxlen=120),
            interpreted_history=deque(maxlen=30),
        )
        self.logger.info(
            "engine_initialized",
            extra={
                "baseline_window": self.config.baseline_window,
                "recent_window": self.config.recent_window,
                "relation_threshold": self.config.relation_threshold,
            },
        )

    @staticmethod
    def _to_float_timestamp(value: str, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(fallback)

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
        if vals.size < 4:
            return 0.0
        x = np.arange(vals.size, dtype=float)
        slope, _ = np.polyfit(x, vals, 1)
        return float(slope)

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
                "graph_metrics": {},
                "regime": {},
                "context": None,
            },
        }
        return out

    def _update_adaptive_baseline(self, recent_corr: np.ndarray, state: str) -> None:
        if self.state.baseline_corr is None:
            self.state.baseline_corr = np.asarray(recent_corr, dtype=float)
            return
        if self.state.processed_frames < self.config.freeze_baseline_frames:
            return
        if state != "STABLE":
            return
        alpha = float(self.config.baseline_adaptation_alpha)
        self.state.baseline_corr = alpha * self.state.baseline_corr + (1.0 - alpha) * recent_corr
        adapted_adj = (np.abs(self.state.baseline_corr) >= float(self.config.relation_threshold)).astype(float)
        np.fill_diagonal(adapted_adj, 0.0)
        self.state.baseline_adj = adapted_adj

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
                threshold=self.config.relation_threshold,
            )

            if self.state.baseline_corr is None:
                self.state.baseline_corr = np.asarray(geom.baseline_corr, dtype=float)
            if self.state.baseline_adj is None:
                self.state.baseline_adj = np.asarray(gstate.adjacency, dtype=float)

            reg_assign = self.regimes.observe(
                RegimeObservation(
                    signature=np.asarray(np.concatenate([geom.recent_mean, geom.recent_std]), dtype=float),
                    graph_signature=np.asarray(
                        [float(gstate.density), float(gstate.avg_degree), float(gstate.l1_deformation)],
                        dtype=float,
                    ),
                )
            )
            regime_distance = (
                float(reg_assign.regime_distance)
                if reg_assign.regime_distance is not None
                else float(self.config.regime_distance_threshold)
            )

            coherence = float(1.0 / (1.0 + max(0.0, geom.relational_instability)))
            coupling = float(max(0.0, geom.relational_instability * (1.0 + gstate.l1_deformation)))
            components = {
                "structural_drift_score": float(geom.structural_drift),
                "relational_instability_score": float(geom.relational_instability),
                "regime_distance": regime_distance,
                "coherence_loss_score": float(1.0 - coherence),
                "graph_deformation_score": float(gstate.l1_deformation),
                "coupling_instability_score": coupling,
            }
            indicators = StructuralIndicators(
                structural_drift_score=float(components["structural_drift_score"]),
                relational_instability_score=float(components["relational_instability_score"]),
                regime_distance=float(components["regime_distance"]),
                coherence_loss_score=float(components["coherence_loss_score"]),
                graph_deformation_score=float(components["graph_deformation_score"]),
                coupling_instability_score=float(components["coupling_instability_score"]),
            )
            composite = self.scoring.composite_departure_score(indicators)
            self.state.composite_history.append(float(composite))

            interpreted = map_to_interpreted_state(
                structural_drift=float(geom.structural_drift),
                relational_instability=float(geom.relational_instability),
                regime_distance=regime_distance,
                coherence_score=coherence,
                graph_instability=float(gstate.l1_deformation),
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
            self._update_adaptive_baseline(geom.recent_corr, decision_state)

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

            driver_names = dominant_drivers(components, top_k=3)
            context = None
            if self.config.allow_context_provider:
                snap = self.context_provider.snapshot(
                    {
                        "timestamp": frame.timestamp,
                        "site_id": frame.site_id,
                        "asset_id": frame.asset_id,
                        "sensor_values": frame.sensor_values,
                    }
                )
                context = None if snap is None else {"source": snap.source, "payload": snap.payload}

            system_health = int(max(0.0, min(100.0, 100.0 - (float(composite) * 35.0))))
            out: SIIResult = {
                "timestamp": frame.timestamp,
                "site_id": frame.site_id,
                "asset_id": frame.asset_id,
                "state": decision_state,
                "interpreted_state": interpreted,
                "confidence": confidence,
                "structural_drift_score": round(float(geom.structural_drift), 4),
                "relational_instability_score": round(float(geom.relational_instability), 4),
                "regime_distance": round(float(regime_distance), 4),
                "coherence_score": round(float(coherence), 4),
                "graph_deformation_score": round(float(gstate.l1_deformation), 4),
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
                    "composite_instability": round(float(composite), 6),
                    "graph_metrics": {
                        "density": round(float(gstate.density), 6),
                        "avg_degree": round(float(gstate.avg_degree), 6),
                        "deformation": round(float(gstate.l1_deformation), 6),
                    },
                    "regime": {
                        "name": reg_assign.regime_name,
                        "distance": reg_assign.regime_distance,
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
            raise SIIProcessingError("Failed to process telemetry frame safely") from exc

    def process_payload(self, payload: dict[str, Any]) -> SIIResult:
        f = frame_from_payload(payload)
        return self.process_frame(f)

    def process_csv_text(self, csv_text: str) -> list[SIIResult]:
        frames = frames_from_csv(csv_text)
        return [self.process_frame(f) for f in frames]


SIIEngine = SystemicInfrastructureIntelligenceEngine

