"""Atomic layer detector wrapper for FD00x experiments.

This replaces the previous structural-only detector stack with an Atomic Layer
monitor while preserving the existing evaluation pipeline API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .atomic_calibrated import AtomicMonitorCalibrated
from .atomic_monitor import AtomicMonitor
from .atomic_baseline import AtomicBaseline
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
    directional_proxy: np.ndarray
    latent_centroids: np.ndarray
    latent_occupancy: np.ndarray
    mode_eigenvalues: np.ndarray
    mode_basis: np.ndarray
    event_rate: np.ndarray
    event_interval_cv: np.ndarray
    n_samples: int
    component_score_stats: Dict[str, Any] = field(default_factory=dict)


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
    warning_state: np.ndarray
    warning_enter_threshold: np.ndarray
    warning_exit_threshold: np.ndarray
    component_activation_rates: Dict[str, float]
    dominant_alert_component: str


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
            directional_proxy=b.directional_proxy,
            latent_centroids=b.latent_centroids,
            latent_occupancy=b.latent_occupancy,
            mode_eigenvalues=b.mode_eigenvalues,
            mode_basis=b.mode_basis,
            event_rate=b.event_rate,
            event_interval_cv=b.event_interval_cv,
            n_samples=healthy_data.shape[0],
            component_score_stats=dict(monitor._component_score_stats),
        )

    def score_unit(
        self,
        data: np.ndarray,
        ref: ReferenceStats,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        sensors = [f"s{i}" for i in range(data.shape[1])]
        monitor = self._build_monitor(sensors)
        monitor.baseline = self._baseline_from_reference(ref)
        monitor.online_mu = ref.sde_mu.copy()
        monitor.online_sigma = np.maximum(ref.sde_sigma.copy(), 1e-6)
        # Restore per-component normalization stats from the reference
        if ref.component_score_stats:
            monitor._component_score_stats = dict(ref.component_score_stats)

        raw_scores: List[float] = []
        component: Dict[str, List[float]] = {}

        for t in range(data.shape[0]):
            upd = monitor.update(data[t], timestamp=float(t), context={}) if isinstance(monitor, AtomicMonitorCalibrated) else monitor.update(data[t], timestamp=float(t))
            raw_scores.append(float(upd.score))
            for k, v in upd.components.items():
                component.setdefault(k, []).append(float(v))

        raw = np.asarray(raw_scores, dtype=float)
        ema = _apply_ema(
            raw,
            self.config.ema_alpha,
            adaptive=self.config.adaptive_ema,
            adaptive_window=self.config.adaptive_ema_window,
            min_alpha=self.config.adaptive_ema_min_alpha,
            max_alpha=self.config.adaptive_ema_max_alpha,
            volatility_gain=self.config.adaptive_ema_volatility_gain,
        )

        healthy_ema = ema[: min(ref.n_samples, ema.shape[0])]
        threshold_std = override_threshold_std if override_threshold_std is not None else self.config.threshold_std
        persistence = override_persistence if override_persistence is not None else self.config.persistence
        static_threshold = _compute_ema_threshold(
            healthy_ema=healthy_ema,
            threshold_mode=self.config.threshold_mode,
            threshold_std=threshold_std,
            threshold_percentile=self.config.threshold_percentile,
        )
        enter_threshold = _compute_dynamic_thresholds(
            ema=ema,
            static_threshold=static_threshold,
            threshold_mode=self.config.threshold_mode,
            threshold_std=threshold_std,
            threshold_percentile=self.config.threshold_percentile,
            rolling_window=self.config.dynamic_threshold_window,
            min_history=self.config.dynamic_threshold_min_history,
        )
        warning_state = compute_warning_state(
            ema,
            enter_threshold,
            persistence,
            require_upward_trend=self.config.require_upward_ema_trend,
            slope_window=self.config.slope_window,
            min_slope=self.config.min_slope,
            exit_threshold_ratio=self.config.warning_exit_threshold_ratio,
            exit_persistence=self.config.exit_persistence,
            min_anomaly_duration=max(self.config.min_anomaly_duration, persistence),
        )
        warning_index = _first_true_index(warning_state)
        component_activation_rates = {
            k: float(np.mean(np.asarray(v, dtype=float) >= self.config.fusion_activation_floor))
            for k, v in component.items()
        }
        dominant_alert_component = (
            max(component_activation_rates, key=component_activation_rates.get)
            if component_activation_rates
            else "none"
        )

        return UnitScores(
            raw_drift=raw,
            ema_drift=ema,
            threshold=float(static_threshold),
            warning_index=warning_index,
            n_cycles=data.shape[0],
            reference_stats=ref,
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
            warning_state=warning_state.astype(bool),
            warning_enter_threshold=enter_threshold.astype(float),
            warning_exit_threshold=(enter_threshold * float(self.config.warning_exit_threshold_ratio)).astype(float),
            component_activation_rates=component_activation_rates,
            dominant_alert_component=dominant_alert_component,
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
            "score_ema_alpha": self.config.score_ema_alpha,
            "fusion_activation_floor": self.config.fusion_activation_floor,
            "fusion_min_active": self.config.fusion_min_active,
            "fusion_downweight_factor": self.config.fusion_downweight_factor,
            "component_weight_multipliers": self.config.component_weight_multipliers,
            "component_soft_caps": self.config.component_soft_caps,
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

    def _baseline_from_reference(self, ref: ReferenceStats) -> AtomicBaseline:
        return AtomicBaseline(
            mean=ref.mean.copy(),
            cov=ref.cov.copy(),
            precision=ref.precision.copy(),
            sde_mu=ref.sde_mu.copy(),
            sde_sigma=ref.sde_sigma.copy(),
            dependence=ref.dependence.copy(),
            directional_proxy=ref.directional_proxy.copy(),
            latent_centroids=ref.latent_centroids.copy(),
            latent_occupancy=ref.latent_occupancy.copy(),
            mode_eigenvalues=ref.mode_eigenvalues.copy(),
            mode_basis=ref.mode_basis.copy(),
            event_rate=ref.event_rate.copy(),
            event_interval_cv=ref.event_interval_cv.copy(),
        )


def find_warning_index(
    scores: np.ndarray,
    threshold: float | np.ndarray,
    persistence: int,
    require_upward_trend: bool = False,
    slope_window: int = 3,
    min_slope: float = 0.0,
    exit_threshold_ratio: float = 0.85,
    exit_persistence: int = 2,
    min_anomaly_duration: int = 1,
) -> Optional[int]:
    state = compute_warning_state(
        scores=scores,
        threshold=threshold,
        persistence=persistence,
        require_upward_trend=require_upward_trend,
        slope_window=slope_window,
        min_slope=min_slope,
        exit_threshold_ratio=exit_threshold_ratio,
        exit_persistence=exit_persistence,
        min_anomaly_duration=min_anomaly_duration,
    )
    return _first_true_index(state)


def compute_warning_state(
    scores: np.ndarray,
    threshold: float | np.ndarray,
    persistence: int,
    require_upward_trend: bool = False,
    slope_window: int = 3,
    min_slope: float = 0.0,
    exit_threshold_ratio: float = 0.85,
    exit_persistence: int = 2,
    min_anomaly_duration: int = 1,
) -> np.ndarray:
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")
    if exit_persistence < 1:
        raise ValueError(f"exit_persistence must be >= 1, got {exit_persistence}")
    if min_anomaly_duration < 1:
        raise ValueError(f"min_anomaly_duration must be >= 1, got {min_anomaly_duration}")

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)
    thr_arr = np.asarray(threshold, dtype=float)
    if thr_arr.ndim == 0:
        enter_thr = np.full(arr.shape[0], float(thr_arr), dtype=float)
    else:
        enter_thr = thr_arr
        if enter_thr.shape[0] != arr.shape[0]:
            raise ValueError("threshold array must match scores length")

    state = np.zeros(arr.shape[0], dtype=bool)
    in_warning = False
    consecutive = 0
    below_exit = 0
    run_start: Optional[int] = None

    for i, s in enumerate(scores):
        val = float(s)
        thr = float(enter_thr[i])
        exit_thr = thr * float(exit_threshold_ratio)
        instant_slope = val - float(arr[i - 1]) if i > 0 else 0.0
        upward_ok = (not require_upward_trend) or (i > 0 and instant_slope > 0.0)
        if min_slope <= 0.0:
            strong_trend = True
        else:
            strong_trend = i >= slope_window and (val - float(arr[i - slope_window])) > min_slope

        if not in_warning:
            if val >= thr:
                if consecutive == 0:
                    if upward_ok and strong_trend:
                        consecutive = 1
                        run_start = i
                else:
                    consecutive += 1
                if consecutive >= persistence and run_start is not None:
                    if (i - run_start + 1) >= min_anomaly_duration:
                        in_warning = True
                        below_exit = 0
                        state[i] = True
            else:
                consecutive = 0
                run_start = None
        else:
            state[i] = True
            if val < exit_thr:
                below_exit += 1
                if below_exit >= exit_persistence:
                    in_warning = False
                    below_exit = 0
                    consecutive = 0
                    run_start = None
                    state[i] = False
            else:
                below_exit = 0
    return state


def _apply_ema(
    series: np.ndarray,
    alpha: float,
    *,
    adaptive: bool = False,
    adaptive_window: int = 15,
    min_alpha: float = 0.08,
    max_alpha: float = 0.30,
    volatility_gain: float = 1.0,
) -> np.ndarray:
    if len(series) == 0:
        return series
    ema = np.empty_like(series)
    ema[0] = series[0]
    if not adaptive:
        use_alpha = float(alpha)
        for i in range(1, len(series)):
            ema[i] = use_alpha * series[i] + (1.0 - use_alpha) * ema[i - 1]
        return ema

    base_alpha = float(alpha)
    local_vol_hist: list[float] = []
    for i in range(1, len(series)):
        lo = max(0, i - max(2, int(adaptive_window)))
        recent = series[lo : i + 1]
        local_vol = float(np.std(recent)) if recent.size > 1 else 0.0
        local_vol_hist.append(local_vol)
        baseline_vol = float(np.median(local_vol_hist)) if local_vol_hist else local_vol
        vol_ratio = local_vol / max(baseline_vol, 1e-6)
        damp = 1.0 + float(volatility_gain) * max(0.0, vol_ratio - 1.0)
        alpha_t = float(np.clip(base_alpha / damp, min_alpha, max_alpha))
        ema[i] = alpha_t * series[i] + (1.0 - alpha_t) * ema[i - 1]
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


def _compute_dynamic_thresholds(
    ema: np.ndarray,
    static_threshold: float,
    threshold_mode: str,
    threshold_std: float,
    threshold_percentile: float,
    rolling_window: int,
    min_history: int,
) -> np.ndarray:
    arr = np.asarray(ema, dtype=float)
    thr = np.full(arr.shape[0], float(static_threshold), dtype=float)
    if rolling_window <= 0:
        return thr
    for i in range(arr.shape[0]):
        if i < int(min_history):
            continue
        lo = max(0, i - int(rolling_window))
        hist = arr[lo:i]
        if hist.size < max(3, int(min_history)):
            continue
        thr[i] = _compute_ema_threshold(
            healthy_ema=hist,
            threshold_mode=threshold_mode,
            threshold_std=threshold_std,
            threshold_percentile=threshold_percentile,
        )
    return thr


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    return int(idx[0]) if idx.size > 0 else None
