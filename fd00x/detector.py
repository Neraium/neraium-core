"""Structural Drift Detector for FD00x evaluation.

Pure structural detection without experimental QIT layer.

Uses:
- Raw structural drift: covariance, correlation, Mahalanobis distance
- EMA smoothing for stability
- Trajectory acceleration: 2nd derivative (degradation curvature)
- Relational instability: correlation breakdown (systemic degradation)
- Multi-signal confirmation: combine amplitude + structural signals

No future data leakage. No RUL in detection logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DetectorConfig
from .structural_signals import StructuralSignalDetector


@dataclass
class ReferenceStats:
    """Frozen reference statistics derived from the healthy segment."""

    mean: np.ndarray
    cov: np.ndarray
    precision: np.ndarray
    std: np.ndarray
    corr: np.ndarray
    n_samples: int
    baseline_data: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    # Legacy fields for API compatibility
    sde_mu: np.ndarray = field(default_factory=lambda: np.array([]))
    sde_sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    dependence: np.ndarray = field(default_factory=lambda: np.array([]))
    directional_proxy: np.ndarray = field(default_factory=lambda: np.array([]))
    latent_centroids: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=float))
    latent_occupancy: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=float))
    mode_eigenvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    mode_basis: np.ndarray = field(default_factory=lambda: np.eye(1, dtype=float))
    event_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    event_interval_cv: np.ndarray = field(default_factory=lambda: np.array([]))
    component_score_stats: Dict[str, Any] = field(default_factory=dict)


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
    """Genuine structural drift detector without experimental layers.

    Detects degradation through:
    1. Raw Structural Drift: Mahalanobis distance, covariance shift, correlation change
    2. Trajectory Acceleration: 2nd derivative (degradation curvature)
    3. Relational Instability: Sensor correlation breakdown
    4. Multi-signal Confirmation: Combine structural + amplitude signals
    5. Persistent Warning State: Lock once confirmed, no oscillation

    Pure structural approach. No RUL information. No future data leakage.
    """

    def __init__(self, config: DetectorConfig, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose
        self.structural_detector = StructuralSignalDetector(verbose=verbose)

    def fit_reference(self, healthy_data: np.ndarray) -> ReferenceStats:
        """Fit reference statistics from healthy (early) segment."""
        healthy = np.asarray(healthy_data, dtype=float)
        if healthy.ndim != 2:
            raise ValueError("healthy_data must be 2D (cycles x sensors)")

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
            std=std,
            corr=corr,
            n_samples=healthy.shape[0],
            baseline_data=healthy.copy(),
            # Legacy API compatibility
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
        )

    def score_unit(
        self,
        data: np.ndarray,
        ref: ReferenceStats,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        """Compute structural drift scores and detect warning."""
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("data must be 2D (cycles x sensors)")

        # Compute raw structural drift at each cycle
        raw = self._compute_structural_drift(arr, ref)

        # Apply EMA smoothing
        ema = _apply_ema(raw, self.config.ema_alpha)

        # Compute component scores (structural components, not QIT)
        component_arrays = self._compute_structural_components(arr, raw, ema, ref)

        # Multi-signal detection with confirmation
        warning_index, warning_state = self._detect_adaptive_warning(ema, raw, arr)

        # Threshold: maximum EMA in healthy region
        threshold = float(np.max(ema[: max(1, min(ref.n_samples, ema.size))]))

        # Component activation rates (for compatibility)
        component_activation_rates = {
            k: float(np.mean(v >= self.config.fusion_activation_floor))
            for k, v in component_arrays.items()
        }
        dominant_alert_component = max(component_activation_rates, key=component_activation_rates.get)

        # Alert history: track state changes
        alert_history = []
        prev_state = False
        for t, current_state in enumerate(warning_state):
            if current_state and not prev_state:
                alert_history.append(
                    {
                        "timestamp": float(t),
                        "level": "warning",
                        "score": float(ema[t]),
                        "dominant_detector": dominant_alert_component,
                    }
                )
            prev_state = current_state

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
        """End-to-end processing: fit reference, compute scores, detect warning."""
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

    def _compute_structural_drift(self, data: np.ndarray, ref: ReferenceStats) -> np.ndarray:
        """Compute raw structural drift score at each cycle.

        Combines:
        - Mahalanobis distance (mean shift)
        - Covariance shift (structure change)
        - Correlation change (relationship change)
        """
        n_cycles = data.shape[0]
        drift = np.zeros(n_cycles)

        window = 15  # Rolling window for recent statistics
        for t in range(n_cycles):
            window_start = max(0, t - window)
            window_data = data[window_start : t + 1]

            if window_data.shape[0] < 2:
                drift[t] = 0.0
                continue

            # Mahalanobis distance from reference mean
            delta = window_data[-1] - ref.mean
            mahal = np.sqrt(delta @ ref.precision @ delta)

            # Covariance shift (Frobenius norm)
            if window_data.shape[0] > 1:
                recent_cov = np.cov(window_data.T) + np.eye(window_data.shape[1]) * 1e-6
                cov_diff = recent_cov - ref.cov
                cov_shift = np.linalg.norm(cov_diff, "fro")
            else:
                cov_shift = 0.0

            # Correlation change
            if window_data.shape[0] > 1:
                recent_corr = np.corrcoef(window_data.T)
                recent_corr = np.nan_to_num(recent_corr, nan=0.0)
                corr_diff = recent_corr - ref.corr
                corr_shift = np.linalg.norm(corr_diff, "fro")
            else:
                corr_shift = 0.0

            # Combined score: equal weighting
            drift[t] = (mahal + cov_shift + corr_shift) / 3.0

        return drift

    def _compute_structural_components(
        self, data: np.ndarray, raw: np.ndarray, ema: np.ndarray, ref: ReferenceStats
    ) -> Dict[str, np.ndarray]:
        """Compute individual structural component scores for diagnostics.

        Components:
        - drift: base structural drift signal
        - acceleration: 2nd derivative (curvature)
        - correlation: correlation breakdown
        - change_point: change-point detections
        - confirmation: confirmation signal
        """
        n_cycles = len(raw)

        # Drift component: normalized EMA
        drift_component = ema / (np.max(ema) + 1e-6) if np.max(ema) > 0 else ema

        # Acceleration component: 2nd derivative
        accel_component = np.zeros(n_cycles)
        if n_cycles > 2:
            vel = np.diff(ema)
            accel = np.diff(vel)
            baseline_std = np.std(ema[: min(20, n_cycles)])
            accel_threshold = 0.005 * baseline_std
            for i in range(len(accel)):
                if accel[i] > accel_threshold:
                    accel_component[i + 2] = min(1.0, accel[i] / (accel_threshold + 1e-6))

        # Correlation component: from structural detector
        corr_candidates = self.structural_detector.compute_correlation_breakdown(data)
        correlation_component = np.zeros(n_cycles)
        correlation_component[corr_candidates] = 1.0

        # Change-point component: CUSUM-based
        baseline_std = np.std(ema[: min(20, n_cycles)])
        cusum_threshold = 0.95 * baseline_std
        cusum_positive = np.zeros(n_cycles)
        for i in range(1, n_cycles):
            delta = ema[i] - np.mean(ema[: min(20, n_cycles)])
            cusum_positive[i] = max(0, cusum_positive[i - 1] + delta - baseline_std * 0.10)
        change_point_component = np.minimum(cusum_positive / (cusum_threshold + 1e-6), 1.0)

        # Confirmation component: will be set by warning detection
        confirmation_component = np.zeros(n_cycles)

        return {
            "drift": drift_component,
            "acceleration": accel_component,
            "correlation": correlation_component,
            "change_point": change_point_component,
            "confirmation": confirmation_component,
        }

    def _detect_adaptive_warning(self, ema: np.ndarray, raw: np.ndarray, sensor_data: np.ndarray) -> tuple[Optional[int], np.ndarray]:
        """Three-phase detection: early signal, confirmation, persistence.

        Phase 1: EARLY SIGNAL
        - Detect amplitude changes (CUSUM, velocity, z-score)
        - Detect structural changes (acceleration, correlation)

        Phase 2: CONFIRMATION
        - Require 2 signals, at least 1 structural
        - Confirm over 3-20 cycle window

        Phase 3: PERSISTENCE
        - Once confirmed, lock warning state
        """
        n_cycles = len(ema)
        if n_cycles < 10:
            return None, np.zeros(n_cycles, dtype=bool)

        # Baseline from first 15%
        baseline_end = min(40, max(5, int(n_cycles * 0.15)))
        baseline_mean = float(np.mean(ema[:baseline_end]))
        baseline_std = float(np.std(ema[:baseline_end]))
        baseline_std = max(baseline_std, 1e-6)

        # --- PHASE 1: EARLY SIGNAL ---

        # Amplitude signals
        # CUSUM
        cusum_threshold = 0.95 * baseline_std
        cusum_positive = np.zeros(n_cycles)
        for i in range(1, n_cycles):
            delta = ema[i] - baseline_mean
            cusum_positive[i] = max(0, cusum_positive[i - 1] + delta - baseline_std * 0.10)
        cusum_candidates = np.where(cusum_positive > cusum_threshold)[0]

        # Velocity
        if n_cycles > 3:
            velocity = np.abs(np.diff(ema))
            baseline_vel = velocity[:baseline_end]
            velocity_threshold = np.percentile(baseline_vel, 55) + 0.8 * np.std(baseline_vel)
            velocity_candidates = np.where(velocity > velocity_threshold)[0] + 1
        else:
            velocity_candidates = np.array([], dtype=int)

        # Z-score
        zscore = np.abs(ema - baseline_mean) / (baseline_std + 1e-6)
        zscore_candidates = np.where(zscore > 1.1)[0]

        # Structural signals
        accel_candidates, corr_candidates = self.structural_detector.detect_all_structural_changes(raw, sensor_data, baseline_std)

        # Combine all candidates
        all_candidates = np.concatenate([cusum_candidates, velocity_candidates, zscore_candidates, accel_candidates, corr_candidates])
        phase1_candidates = np.unique(all_candidates)

        # Filter: only after 15% (healthy region boundary)
        min_cycle = int(n_cycles * 0.15)
        phase1_candidates = phase1_candidates[phase1_candidates >= min_cycle]

        # --- PHASE 2: CONFIRMATION ---

        confirmed_idx = None
        if len(phase1_candidates) > 0:
            first_alert = int(phase1_candidates[0])
            look_back = max(0, first_alert - 3)
            look_forward = min(20, n_cycles - first_alert)
            window_end = min(first_alert + look_forward, n_cycles)
            in_window = ema[look_back:window_end]

            # Check for structural signals in window
            accel_in_window = np.any((accel_candidates >= look_back) & (accel_candidates < window_end))
            corr_in_window = np.any((corr_candidates >= look_back) & (corr_candidates < window_end))
            has_structural = accel_in_window or corr_in_window

            # Check for amplitude signals
            window_has_elevation = np.mean(in_window) > baseline_mean * 1.02
            window_has_breach = np.any(in_window >= baseline_mean + 0.5 * baseline_std)
            window_has_trend = np.any(np.diff(in_window) > 0)
            has_amplitude = window_has_elevation or window_has_breach or window_has_trend

            # Confirm if structural OR amplitude present
            if has_structural or has_amplitude:
                confirmed_idx = first_alert

        # --- PHASE 3: PERSISTENCE ---

        warning_state = np.zeros(n_cycles, dtype=bool)
        if confirmed_idx is not None:
            warning_state[confirmed_idx:] = True

        return confirmed_idx, warning_state


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
    """Legacy function for compatibility. Uses compute_warning_state."""
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
    """Legacy function for compatibility."""
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
    """Exponential moving average."""
    if len(series) == 0:
        return series
    ema = np.empty_like(series)
    ema[0] = series[0]
    use_alpha = float(alpha)
    for i in range(1, len(series)):
        ema[i] = use_alpha * series[i] + (1.0 - use_alpha) * ema[i - 1]
    return ema


def _first_true_index(mask: np.ndarray) -> Optional[int]:
    """Return index of first True value."""
    idx = np.where(np.asarray(mask, dtype=bool))[0]
    return int(idx[0]) if idx.size > 0 else None
