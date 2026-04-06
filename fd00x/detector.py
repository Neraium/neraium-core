"""Atomic layer detector wrapper for FD00x experiments.

This replaces the previous structural-only detector stack with an Atomic Layer
monitor while preserving the existing evaluation pipeline API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .atomic_calibrated import AtomicMonitorCalibrated
from .atomic_monitor import AtomicMonitor
from .config import DetectorConfig


@dataclass
class ReferenceStats:
    """Frozen reference statistics derived from the healthy segment."""

    mean: np.ndarray
    cov: np.ndarray
    precision: np.ndarray
    sde_mu: np.ndarray
    sde_sigma: np.ndarray
    dependence: np.ndarray
    latent_occupancy: np.ndarray
    mode_eigenvalues: np.ndarray
    event_rate: np.ndarray
    n_samples: int


@dataclass
class UnitScores:
    """Per-unit atomic anomaly timeline and detection outputs."""

    raw_drift: np.ndarray
    ema_drift: np.ndarray
    threshold: float
    warning_index: Optional[int]
    n_cycles: int
    reference_stats: ReferenceStats
    component_scores: Dict[str, np.ndarray]
    alert_history: List[dict]


class StructuralDriftDetector:
    """Compatibility wrapper exposing the old detector API with atomic internals."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    def fit_reference(self, healthy_data: np.ndarray) -> ReferenceStats:
        sensors = [f"s{i}" for i in range(healthy_data.shape[1])]
        monitor = self._build_monitor(sensors)
        monitor.learn_baseline(healthy_data)
        b = monitor.baseline
        return ReferenceStats(
            mean=b.mean,
            cov=b.cov,
            precision=b.precision,
            sde_mu=b.sde_mu,
            sde_sigma=b.sde_sigma,
            dependence=b.dependence,
            latent_occupancy=b.latent_occupancy,
            mode_eigenvalues=b.mode_eigenvalues,
            event_rate=b.event_rate,
            n_samples=healthy_data.shape[0],
        )

    def score_unit(
        self,
        data: np.ndarray,
        ref: ReferenceStats,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        sensors = [f"s{i}" for i in range(data.shape[1])]
        cfg = self._atomic_config()
        monitor = self._build_monitor(sensors)
        healthy = data[: ref.n_samples]
        monitor.learn_baseline(healthy)
        if isinstance(monitor, AtomicMonitorCalibrated) and self.config.calibration_enabled:
            monitor.calibrate(healthy_data=healthy, validation_data=None)

        raw_scores: List[float] = []
        component: Dict[str, List[float]] = {}

        for t in range(data.shape[0]):
            upd = monitor.update(data[t], timestamp=float(t), context={}) if isinstance(monitor, AtomicMonitorCalibrated) else monitor.update(data[t], timestamp=float(t))
            raw_scores.append(float(upd.score))
            for k, v in upd.components.items():
                component.setdefault(k, []).append(float(v))

        raw = np.asarray(raw_scores, dtype=float)
        ema = _apply_ema(raw, self.config.ema_alpha)

        healthy_ema = ema[: ref.n_samples]
        threshold_std = override_threshold_std if override_threshold_std is not None else self.config.threshold_std
        persistence = override_persistence if override_persistence is not None else self.config.persistence
        threshold = _compute_ema_threshold(
            healthy_ema=healthy_ema,
            threshold_mode=self.config.threshold_mode,
            threshold_std=threshold_std,
            threshold_percentile=self.config.threshold_percentile,
        )
        warning_index = find_warning_index(
            ema,
            threshold,
            persistence,
            require_upward_trend=self.config.require_upward_ema_trend,
            slope_window=self.config.slope_window,
            min_slope=self.config.min_slope,
        )

        ref_out = self.fit_reference(healthy)
        return UnitScores(
            raw_drift=raw,
            ema_drift=ema,
            threshold=threshold,
            warning_index=warning_index,
            n_cycles=data.shape[0],
            reference_stats=ref_out,
            component_scores={k: np.asarray(v, dtype=float) for k, v in component.items()},
            alert_history=[
                {
                    "timestamp": a.timestamp,
                    "level": a.level,
                    "score": a.score,
                    "dominant_detector": a.dominant_detector,
                }
                for a in monitor.alert_machine.history
            ],
        )

    def process_unit(
        self,
        data: np.ndarray,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        n = len(data)
        healthy_end = max(self.config.min_reference_samples, int(n * self.config.healthy_fraction))
        healthy_end = min(healthy_end, n - 1)
        ref = self.fit_reference(data[:healthy_end])
        return self.score_unit(
            data,
            ref,
            override_threshold_std=override_threshold_std,
            override_persistence=override_persistence,
        )

    def _atomic_config(self) -> dict:
        return {
            "window_size": self.config.window_size,
            "forgetting_factor": self.config.forgetting_factor,
            "sde_dt": self.config.sde_dt,
            "te_k": self.config.te_k,
            "latent_state_count": self.config.latent_state_count,
            "dmd_rank": self.config.dmd_rank,
            "compute_intervals": {
                "structure": self.config.compute_interval_structure,
                "state": self.config.compute_interval_state,
                "topology": self.config.compute_interval_topology,
                "events": self.config.compute_interval_events,
            },
            "alert_thresholds": {
                "green_yellow": self.config.green_yellow,
                "yellow_red": self.config.yellow_red,
            },
            "detector_weights": self.config.detector_weights,
            "event_level_std": self.config.event_level_std,
            "conformal_enabled": self.config.conformal_enabled,
            "conformal_alpha": self.config.conformal_alpha,
            "conformal_window": self.config.conformal_window,
            "operational_modes": self.config.operational_modes,
            "maintenance_windows": self.config.maintenance_windows,
            "diurnal_patterns": self.config.diurnal_patterns,
            "consensus_required": self.config.consensus_required,
            "consensus_window": self.config.consensus_window,
            "threshold_adaptation": self.config.threshold_adaptation,
            "target_fp_rate": self.config.target_fp_rate,
            "fp_history_window": self.config.fp_history_window,
            "bh_enabled": self.config.bh_enabled,
            "bh_fdr_target": self.config.bh_fdr_target,
            "min_alert_duration": self.config.min_alert_duration,
            "cooldown_period": self.config.cooldown_period,
        }

    def _build_monitor(self, sensors: List[str]) -> AtomicMonitor:
        cfg = self._atomic_config()
        if self.config.calibration_enabled:
            return AtomicMonitorCalibrated(sensors=sensors, config=cfg)
        return AtomicMonitor(sensors=sensors, config=cfg)


def find_warning_index(
    scores: np.ndarray,
    threshold: float,
    persistence: int,
    require_upward_trend: bool = False,
    slope_window: int = 3,
    min_slope: float = 0.0,
) -> Optional[int]:
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")
    consecutive = 0
    for i, s in enumerate(scores):
        val = float(s)
        instant_slope = val - float(scores[i - 1]) if i > 0 else 0.0
        upward_ok = (not require_upward_trend) or (i > 0 and instant_slope > 0.0)
        if min_slope <= 0.0:
            strong_trend = True
        else:
            strong_trend = i >= slope_window and (val - float(scores[i - slope_window])) > min_slope

        if val >= threshold:
            if consecutive == 0:
                if upward_ok and strong_trend:
                    consecutive = 1
            else:
                consecutive += 1
            if consecutive >= persistence:
                return i
        else:
            consecutive = 0
    return None


def _apply_ema(series: np.ndarray, alpha: float) -> np.ndarray:
    if len(series) == 0:
        return series
    ema = np.empty_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_ema_threshold(
    healthy_ema: np.ndarray,
    threshold_mode: str,
    threshold_std: float,
    threshold_percentile: float,
) -> float:
    if healthy_ema.size == 0:
        return 1e-6
    mode = str(threshold_mode).strip().lower()
    if mode == "mean_std":
        return float(np.mean(healthy_ema) + threshold_std * max(np.std(healthy_ema), 1e-6))
    if mode == "percentile":
        return float(np.percentile(healthy_ema, float(np.clip(threshold_percentile, 0.0, 100.0))))
    if mode == "robust_mad":
        m = float(np.median(healthy_ema))
        mad = float(np.median(np.abs(healthy_ema - m)))
        return m + threshold_std * max(mad, 1e-6)
    raise ValueError(f"Unsupported threshold_mode '{threshold_mode}'")
