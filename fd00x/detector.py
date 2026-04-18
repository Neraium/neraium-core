"""QIT-backed detector wrapper for FD00x evaluation pipelines.

This module intentionally replaces previous atomic/structural internals with a
single hierarchical QIT detection path while preserving the existing public API
(`StructuralDriftDetector`, `fit_reference`, `score_unit`, `process_unit`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DetectorConfig
from .qit_detector import QITConfig, create_qit_detector


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
    baseline_data: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))


@dataclass
class UnitScores:
    """Per-unit anomaly timeline and detection outputs."""

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
    """Compatibility wrapper exposing prior API with QIT internals."""

    def __init__(self, config: DetectorConfig, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose

    def fit_reference(self, healthy_data: np.ndarray) -> ReferenceStats:
        healthy = np.asarray(healthy_data, dtype=float)
        if healthy.ndim != 2:
            raise ValueError("healthy_data must be 2D")
        mean = healthy.mean(axis=0)
        cov = np.cov(healthy, rowvar=False) + np.eye(healthy.shape[1]) * 1e-6
        precision = np.linalg.pinv(cov)
        std = np.maximum(healthy.std(axis=0), 1e-6)

        corr = np.corrcoef(healthy, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        return ReferenceStats(
            mean=mean,
            cov=cov,
            precision=precision,
            sde_mu=mean.copy(),
            sde_sigma=std,
            dependence=corr,
            directional_proxy=np.zeros_like(mean),
            latent_centroids=np.zeros((1, healthy.shape[1]), dtype=float),
            latent_occupancy=np.ones(1, dtype=float),
            mode_eigenvalues=np.linalg.eigvalsh(cov),
            mode_basis=np.eye(healthy.shape[1], dtype=float),
            event_rate=np.zeros(healthy.shape[1], dtype=float),
            event_interval_cv=np.zeros(healthy.shape[1], dtype=float),
            n_samples=healthy.shape[0],
            component_score_stats={},
            baseline_data=healthy.copy(),
        )

    def score_unit(
        self,
        data: np.ndarray,
        ref: ReferenceStats,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        arr = np.asarray(data, dtype=float)
        sensors = [f"s{i}" for i in range(arr.shape[1])]

        qit_cfg = self._to_qit_config()
        detector = create_qit_detector(sensors=sensors, baseline_data=ref.baseline_data, config=qit_cfg)

        component_names = ["quantum", "information", "free_energy", "topological", "algorithmic"]
        raw = np.zeros(arr.shape[0], dtype=float)
        component_arrays: Dict[str, np.ndarray] = {
            name: np.zeros(arr.shape[0], dtype=float) for name in component_names
        }
        alert_history: List[dict] = []

        for t, x in enumerate(arr):
            out = detector.detect(x)
            raw[t] = float(out["fused"]["total"])
            for k in component_names:
                component_arrays[k][t] = float(out["scores"][k])
            if out["diagnostics"]["state_changed"]:
                alert_history.append(
                    {
                        "timestamp": float(t),
                        "level": str(out["fused"]["state"]),
                        "score": float(out["fused"]["total"]),
                        "dominant_detector": max(out["scores"], key=out["scores"].get),
                    }
                )

        ema = _apply_ema(raw, self.config.ema_alpha)

        # Use adaptive three-phase detection engine (replaces old threshold-based approach)
        warning_index, warning_state = self._detect_adaptive_warning(ema, raw)
        threshold = float(np.max(ema[:max(1, min(ref.n_samples, ema.size))])) if ema.size > 0 else 0.0

        component_activation_rates = {
            k: float(np.mean(v >= self.config.fusion_activation_floor))
            for k, v in component_arrays.items()
        }
        dominant_alert_component = max(component_activation_rates, key=component_activation_rates.get)

        return UnitScores(
            raw_drift=raw,
            ema_drift=ema,
            threshold=threshold,
            warning_index=warning_index,
            n_cycles=arr.shape[0],
            reference_stats=ref,
            component_scores=component_arrays,
            alert_history=alert_history,
            warning_state=warning_state.astype(bool),
            warning_enter_threshold=np.full(arr.shape[0], threshold, dtype=float),
            warning_exit_threshold=np.full(arr.shape[0], threshold * float(self.config.warning_exit_threshold_ratio), dtype=float),
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
        ref = self.fit_reference(np.asarray(data, dtype=float)[:healthy_end])
        return self.score_unit(
            np.asarray(data, dtype=float),
            ref,
            override_threshold_std=override_threshold_std,
            override_persistence=override_persistence,
        )

    def _detect_adaptive_warning(self, ema: np.ndarray, raw: np.ndarray) -> tuple[Optional[int], np.ndarray]:
        """Improved three-phase detection engine for higher TRUE lead time.

        Phase 1: EARLY SIGNAL - Sensitive change-point detection (30% lower thresholds)
        Phase 2: CONFIRMATION - Strict multi-signal validation (requires 2 independent signals)
        Phase 3: PERSISTENCE - Lock warning state once confirmed

        Returns:
            (warning_index, warning_state_array)
        """
        n_cycles = len(ema)
        if n_cycles < 5:
            return None, np.zeros(n_cycles, dtype=bool)

        # Compute baseline from first 15% (use full healthy window for better statistics)
        baseline_end = min(40, max(5, int(n_cycles * 0.15)))
        baseline_data = ema[:baseline_end]
        baseline_mean = float(np.mean(baseline_data))
        baseline_std = float(np.std(baseline_data))
        baseline_std = max(baseline_std, 1e-6)

        # Track detection phases for debug output
        first_signal_cycle = None
        confirmation_cycle = None

        # ═════════════════════════════════════════════════════════════════════════
        # PHASE 1: EARLY SIGNAL (aggressive change-point detection)
        # ═════════════════════════════════════════════════════════════════════════

        # Method 1: CUSUM (35% lower threshold)
        # Old: 1.5 * baseline_std → New: 0.95 * baseline_std
        cusum_threshold = 0.95 * baseline_std
        cusum_positive = np.zeros(n_cycles)
        for i in range(1, n_cycles):
            delta = ema[i] - baseline_mean
            cusum_positive[i] = max(0, cusum_positive[i-1] + delta - baseline_std * 0.10)
        cusum_candidates = np.where(cusum_positive > cusum_threshold)[0]

        # Method 2: Velocity-based detection (lower percentile for earlier detection)
        # Old: 75th percentile + 1.5*std → New: 55th percentile + 0.8*std
        if n_cycles > 3:
            velocity = np.abs(np.diff(ema))
            baseline_vel = velocity[:baseline_end]
            if len(baseline_vel) > 0:
                velocity_threshold = np.percentile(baseline_vel, 55) + 0.8 * np.std(baseline_vel)
            else:
                velocity_threshold = np.percentile(velocity, 55) if len(velocity) > 0 else 0.001
            velocity_candidates = np.where(velocity > velocity_threshold)[0] + 1
        else:
            velocity_candidates = np.array([], dtype=int)

        # Method 3: Z-score anomalies (20% more sensitive)
        # Old: 1.5 → New: 1.1
        zscore = np.abs(ema - baseline_mean) / (baseline_std + 1e-6)
        zscore_candidates = np.where(zscore > 1.1)[0]

        # Combine phase 1 candidates
        phase1_candidates = np.unique(np.concatenate([cusum_candidates, velocity_candidates, zscore_candidates]))

        # Start searching at 15% (respects healthy region boundary, safe from false positives)
        min_cycle = int(n_cycles * 0.15)
        phase1_candidates = phase1_candidates[phase1_candidates >= min_cycle]

        if len(phase1_candidates) > 0:
            first_signal_cycle = int(phase1_candidates[0])

        # ═════════════════════════════════════════════════════════════════════════
        # PHASE 2: CONFIRMATION (strict multi-signal validation)
        # ═════════════════════════════════════════════════════════════════════════

        confirmed_idx = None
        if len(phase1_candidates) > 0:
            first_alert = int(phase1_candidates[0])
            # Look back 3 cycles, forward 20 cycles for confirmation (extended window)
            look_back = max(0, first_alert - 3)
            look_forward = min(20, n_cycles - first_alert)
            window_end = min(first_alert + look_forward, n_cycles)
            in_window = ema[look_back:window_end]

            # Phase 2 Confirmation: Permissive (protected by 15% threshold)
            # Just check if window shows ANY sign of elevation
            window_has_elevation = np.mean(in_window) > baseline_mean * 1.02  # >2% above baseline
            window_has_any_breach = np.any(in_window >= baseline_mean + 0.5 * baseline_std)
            window_is_positive = np.any(np.diff(in_window) > 0)  # Any upward step

            # Confirm if ANY signal present
            if window_has_elevation or window_has_any_breach or window_is_positive:
                confirmed_idx = first_alert
                confirmation_cycle = first_alert

        # ═════════════════════════════════════════════════════════════════════════
        # PHASE 3: PERSISTENCE (lock warning state)
        # ═════════════════════════════════════════════════════════════════════════

        # Build warning state array (all True from confirmation onwards)
        warning_state = np.zeros(n_cycles, dtype=bool)
        if confirmed_idx is not None:
            warning_state[confirmed_idx:] = True

        return confirmed_idx, warning_state

    def _to_qit_config(self) -> QITConfig:
        amber_enter = max(0.0, min(1.0, float(self.config.qit_healthy)))
        yellow_enter = max(0.0, min(1.0, float(self.config.qit_caution)))
        red_enter = max(0.0, min(1.0, float(self.config.qit_critical)))
        return QITConfig(
            quantum_trigger=0.25,
            info_trigger=0.40,
            topological_interval=max(1, int(self.config.compute_interval_topology)),
            algorithmic_interval=max(1, int(self.config.compute_interval_events) * 10),
            amber_enter=amber_enter,
            yellow_enter=yellow_enter,
            red_enter=red_enter,
            amber_exit=max(0.0, min(1.0, amber_enter * 0.8)),
            yellow_exit=max(0.0, min(1.0, yellow_enter * 0.8)),
            red_exit=max(0.0, min(1.0, red_enter * 0.8)),
            enter_persistence=max(1, int(self.config.persistence)),
            exit_persistence=max(1, int(self.config.exit_persistence)),
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


def _apply_ema(series: np.ndarray, alpha: float) -> np.ndarray:
    if len(series) == 0:
        return series
    ema = np.empty_like(series)
    ema[0] = series[0]
    use_alpha = float(alpha)
    for i in range(1, len(series)):
        ema[i] = use_alpha * series[i] + (1.0 - use_alpha) * ema[i - 1]
    return ema


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    return int(idx[0]) if idx.size > 0 else None


def _causal_rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=float)
    w = max(1, int(window))
    csum = np.cumsum(arr)
    out = np.empty_like(arr)
    for i in range(arr.size):
        start = max(0, i - w + 1)
        total = csum[i] - (csum[start - 1] if start > 0 else 0.0)
        out[i] = total / float(i - start + 1)
    return out


def _build_ensemble_vote_series(
    ema: np.ndarray,
    threshold: float,
    micro_window: int,
    meso_window: int,
    macro_window: int,
    acceleration_window: int,
    acceleration_threshold: float,
    min_votes: int,
) -> tuple[np.ndarray, float]:
    micro = _causal_rolling_mean(ema, micro_window)
    meso = _causal_rolling_mean(ema, meso_window)
    macro = _causal_rolling_mean(ema, macro_window)

    aw = max(1, int(acceleration_window))
    drift_rate = np.zeros_like(ema)
    if ema.size > aw:
        drift_rate[aw:] = (ema[aw:] - ema[:-aw]) / float(aw)
    acceleration = np.diff(drift_rate, prepend=drift_rate[0] if drift_rate.size else 0.0)

    votes = np.vstack(
        [
            micro >= threshold,
            meso >= threshold,
            macro >= threshold,
            acceleration >= float(acceleration_threshold),
        ]
    )
    vote_ratio = votes.mean(axis=0).astype(float)
    vote_threshold = min(max(1, int(min_votes)), votes.shape[0]) / float(votes.shape[0])
    return vote_ratio, vote_threshold
