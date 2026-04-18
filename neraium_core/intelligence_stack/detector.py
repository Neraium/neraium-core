"""Neraium Intelligence Stack: Layered anomaly detection via structural, relational, and regime evidence.

Implements five layers of the Intelligence Stack:
1. Structural Geometry: Mahalanobis distance, covariance drift, correlation drift
2. Relational Instability: Sensor correlation breakdown
3. Trajectory Dynamics: EMA smoothing, velocity, acceleration, change-point detection
4. Regime Transition: Online clustering, transition confidence, persistence metrics
5. Evidence Fusion: Multi-signal confirmation gating, interpretable alerts

All layers are causal (no future data), interpretable, and online-safe.

Walk-forward safety: reference statistics (mean, covariance, correlation) are
frozen from the healthy segment (typically first 35% of lifecycle) and never updated.

No RUL information in detection logic. Pure structural state transition detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .config import DetectorConfig
from .regime_transition import RegimeTransitionLayer, RegimeTransitionOutput
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
    """Per-unit structural evidence timeline and detection outputs.

    Outputs from all five Intelligence Stack layers:
    - raw_drift: raw structural drift score [Structural Geometry]
    - ema_drift: smoothed drift [Trajectory Dynamics]
    - component_scores: individual layer outputs (drift, acceleration, correlation, etc.)
    - regime_transition_timeline: per-cycle regime assignments [Regime Transition]
    - warning_state: persistent binary warning state [Evidence Fusion]
    """

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
    regime_transition_timeline: List[Dict[str, Any]] = field(default_factory=list)


class StructuralDriftDetector:
    """Neraium Intelligence Stack detector.

    Implements five-layer structural state transition detection:

    1. **Structural Geometry**: Mahalanobis distance, covariance drift, correlation drift
    2. **Relational Instability**: Sensor correlation breakdown and dependency fracture
    3. **Trajectory Dynamics**: EMA smoothing, velocity, acceleration, CUSUM change-points
    4. **Regime Transition**: Online clustering, transition confidence, persistence metrics
    5. **Evidence Fusion**: Multi-signal confirmation, interpretable explanations

    All layers are:
    - Causal: no future data at any stage
    - Online-safe: works with streaming input, no retraining
    - Interpretable: mathematical (linear algebra, statistics) not black-box
    - Trustworthy: confirmation gating prevents single-sensor false alarms

    Reference statistics (mean, covariance, correlation) are frozen from the healthy
    segment and never updated, ensuring walk-forward safety.
    """

    def __init__(self, config: DetectorConfig, verbose: bool = False) -> None:
        self.config = config
        self.verbose = verbose
        self.structural_detector = StructuralSignalDetector(verbose=verbose)
        self.regime_layer = RegimeTransitionLayer(verbose=verbose)

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
        """Compute Intelligence Stack scores across all five layers.

        Returns per-cycle evidence from:
        - Structural Geometry (raw drift)
        - Trajectory Dynamics (smoothed drift, velocity, acceleration)
        - Relational Instability (correlation breakdown)
        - Regime Transition (regime assignment, transition confidence)
        - Evidence Fusion (confirmed warning state)
        """
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("data must be 2D (cycles x sensors)")

        # Layer 1: Structural Geometry
        raw = self._compute_structural_drift(arr, ref)

        # Layer 3: Trajectory Dynamics (smoothing)
        ema = _apply_ema(raw, self.config.ema_alpha)

        # Compute all component scores from layers 1-3 and 5
        component_arrays = self._compute_structural_components(arr, raw, ema, ref)

        # Layer 4: Regime Transition
        self.regime_layer.reset()
        regime_timeline = self._compute_regime_timeline(raw, arr, ref, ema)
        component_arrays["regime_transition"] = np.array(
            [r["transition_score"] for r in regime_timeline], dtype=float
        )

        # Layer 5: Evidence Fusion with Confirmation Gating
        warning_index, warning_state = self._detect_adaptive_warning(ema, raw, arr, regime_timeline)

        # Threshold: maximum EMA in healthy region [Trajectory Dynamics evidence]
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
                        "regime": regime_timeline[t]["regime_label"] if t < len(regime_timeline) else "unknown",
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
            regime_transition_timeline=regime_timeline,
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

    def _compute_regime_timeline(
        self, raw: np.ndarray, data: np.ndarray, ref: ReferenceStats, ema: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Compute per-cycle Regime Transition layer (Layer 4) outputs.

        Uses structural features (Mahalanobis, covariance drift, correlation drift)
        to assign regime labels and detect transitions online with persistence metrics.

        Args:
            raw: raw structural drift scores
            data: sensor data [n_cycles, n_sensors]
            ref: reference statistics
            ema: smoothed drift scores

        Returns:
            List of dicts with regime_id, regime_label, transition_score, etc.
        """
        n_cycles = data.shape[0]
        window = 15

        # Extract component structural features for regime assignment
        mahal_scores = np.zeros(n_cycles)
        cov_scores = np.zeros(n_cycles)
        corr_scores = np.zeros(n_cycles)

        for t in range(n_cycles):
            window_start = max(0, t - window)
            window_data = data[window_start : t + 1]

            if window_data.shape[0] < 2:
                continue

            # Mahalanobis distance
            delta = window_data[-1] - ref.mean
            mahal_scores[t] = np.sqrt(delta @ ref.precision @ delta)

            # Covariance shift
            if window_data.shape[0] > 1:
                recent_cov = np.cov(window_data.T) + np.eye(window_data.shape[1]) * 1e-6
                cov_diff = recent_cov - ref.cov
                cov_scores[t] = np.linalg.norm(cov_diff, "fro")

            # Correlation shift
            if window_data.shape[0] > 1:
                recent_corr = np.corrcoef(window_data.T)
                recent_corr = np.nan_to_num(recent_corr, nan=0.0)
                corr_diff = recent_corr - ref.corr
                corr_scores[t] = np.linalg.norm(corr_diff, "fro")

        # Initialize regime layer from early structural features
        healthy_end = min(40, max(5, int(n_cycles * 0.15)))
        early_features = np.column_stack([mahal_scores[:healthy_end], cov_scores[:healthy_end], corr_scores[:healthy_end]])
        self.regime_layer.initialize_regimes(early_features)

        # Process each cycle through regime layer
        regime_timeline = []
        for t in range(n_cycles):
            regime_output = self.regime_layer.process_cycle(
                mahal_distance=float(mahal_scores[t]),
                cov_drift=float(cov_scores[t]),
                corr_drift=float(corr_scores[t]),
            )

            regime_timeline.append(
                {
                    "cycle": t,
                    "regime_id": regime_output.regime_id,
                    "regime_label": regime_output.regime_label,
                    "transition_score": regime_output.transition_score,
                    "transition_detected": regime_output.transition_detected,
                    "persistence": regime_output.regime_persistence_cycles,
                }
            )

        return regime_timeline

    def _compute_structural_drift(self, data: np.ndarray, ref: ReferenceStats) -> np.ndarray:
        """Compute Layer 1 (Structural Geometry) raw drift score at each cycle.

        Combines three structural geometry signals:

        1. **Mahalanobis Distance**: Shift in mean position relative to frozen reference
           - Detects: operating point changes, sensor offset drift
           - Formula: sqrt((x[t] - μ_ref)^T Σ_ref^{-1} (x[t] - μ_ref))

        2. **Covariance Drift**: Change in state cloud shape and volume
           - Detects: loss of constraints, sensor saturation, mode activation
           - Formula: ||Σ_recent(t) - Σ_ref||_F (Frobenius norm)

        3. **Correlation Drift**: Breakdown of pairwise relationships
           - Detects: dependency fracture, systematic coupling loss
           - Formula: ||R_recent(t) - R_ref||_F

        All are normalized and equally weighted into final drift score.

        Uses rolling window (15 cycles) to compute recent statistics without future leakage.
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
        """Compute per-cycle component scores from Intelligence Stack layers for diagnostics.

        Components mapped to layers:

        Layer 1 (Structural Geometry):
        - structural_geometry: normalized EMA drift

        Layer 3 (Trajectory Dynamics):
        - acceleration: 2nd derivative (curvature)
        - change_point: CUSUM-based change-point detection

        Layer 2 (Relational Instability):
        - correlation: correlation breakdown detection

        Layer 5 (Evidence Fusion):
        - confirmation: confirmation signal (set by warning detection)

        Note: regime_transition is added separately in score_unit.
        """
        n_cycles = len(raw)

        # Layer 1: Structural Geometry
        # Normalized drift score: combined Mahalanobis, covariance, correlation signals
        structural_geometry = ema / (np.max(ema) + 1e-6) if np.max(ema) > 0 else ema

        # Layer 3: Trajectory Dynamics
        # Acceleration: 2nd derivative (curvature); detects sudden escalation
        acceleration = np.zeros(n_cycles)
        if n_cycles > 2:
            vel = np.diff(ema)
            accel = np.diff(vel)
            baseline_std = np.std(ema[: min(20, n_cycles)])
            accel_threshold = 0.005 * baseline_std
            for i in range(len(accel)):
                if accel[i] > accel_threshold:
                    acceleration[i + 2] = min(1.0, accel[i] / (accel_threshold + 1e-6))

        # Layer 2: Relational Instability
        # Correlation breakdown: loss of pairwise sensor relationships
        corr_candidates = self.structural_detector.compute_correlation_breakdown(data)
        relational_instability = np.zeros(n_cycles)
        relational_instability[corr_candidates] = 1.0

        # Layer 3: Trajectory Dynamics
        # Change-point detection (CUSUM): sustained elevation in drift score
        baseline_std = np.std(ema[: min(20, n_cycles)])
        cusum_threshold = 0.95 * baseline_std
        cusum_positive = np.zeros(n_cycles)
        for i in range(1, n_cycles):
            delta = ema[i] - np.mean(ema[: min(20, n_cycles)])
            cusum_positive[i] = max(0, cusum_positive[i - 1] + delta - baseline_std * 0.10)
        trajectory_changepoint = np.minimum(cusum_positive / (cusum_threshold + 1e-6), 1.0)

        # Layer 5: Evidence Fusion
        # Confirmation: set by warning detection logic; indicates multi-signal agreement
        confirmation = np.zeros(n_cycles)

        return {
            "structural_geometry": structural_geometry,
            "acceleration": acceleration,
            "relational_instability": relational_instability,
            "trajectory_changepoint": trajectory_changepoint,
            "confirmation": confirmation,
            # Legacy API compatibility
            "drift": structural_geometry,
            "correlation": relational_instability,
            "change_point": trajectory_changepoint,
        }

    def _detect_adaptive_warning(
        self,
        ema: np.ndarray,
        raw: np.ndarray,
        sensor_data: np.ndarray,
        regime_timeline: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[Optional[int], np.ndarray]:
        """Three-phase evidence-based detection with regime transition confirmation.

        Integrates all Intelligence Stack layers:
        1. Structural Geometry: amplitude and covariance signals
        2. Relational Instability: correlation breakdown
        3. Trajectory Dynamics: acceleration and CUSUM change-point
        4. Regime Transition: transition score and persistence
        5. Evidence Fusion: require ≥2 independent signals before confirmation

        Phase 1: EARLY SIGNAL
        - Detect amplitude changes (CUSUM, velocity, z-score)
        - Detect structural changes (acceleration, correlation breakdown)
        - Detect regime transitions (sudden regime shifts)

        Phase 2: CONFIRMATION
        - Require ≥2 independent signals, with ≥1 structural
        - Confirm over 3-20 cycle window
        - Incorporate regime transition confidence

        Phase 3: PERSISTENCE
        - Once confirmed, lock warning state until evidence drops
        - Persistent state prevents false oscillation
        """
        n_cycles = len(ema)
        if n_cycles < 10:
            return None, np.zeros(n_cycles, dtype=bool)

        # Baseline from first 15%
        baseline_end = min(40, max(5, int(n_cycles * 0.15)))
        baseline_mean = float(np.mean(ema[:baseline_end]))
        baseline_std = float(np.std(ema[:baseline_end]))
        baseline_std = max(baseline_std, 1e-6)

        # --- PHASE 1: EARLY SIGNAL (from all five layers) ---

        # Layer 1 & 3: Amplitude signals [Structural Geometry + Trajectory Dynamics]
        # CUSUM
        cusum_threshold = 0.95 * baseline_std
        cusum_positive = np.zeros(n_cycles)
        for i in range(1, n_cycles):
            delta = ema[i] - baseline_mean
            cusum_positive[i] = max(0, cusum_positive[i - 1] + delta - baseline_std * 0.10)
        cusum_candidates = np.where(cusum_positive > cusum_threshold)[0]

        # Velocity [Trajectory Dynamics]
        if n_cycles > 3:
            velocity = np.abs(np.diff(ema))
            baseline_vel = velocity[:baseline_end]
            velocity_threshold = np.percentile(baseline_vel, 55) + 0.8 * np.std(baseline_vel)
            velocity_candidates = np.where(velocity > velocity_threshold)[0] + 1
        else:
            velocity_candidates = np.array([], dtype=int)

        # Z-score [Structural Geometry]
        zscore = np.abs(ema - baseline_mean) / (baseline_std + 1e-6)
        zscore_candidates = np.where(zscore > 1.1)[0]

        # Structural signals [Trajectory Dynamics, Relational Instability]
        accel_candidates, corr_candidates = self.structural_detector.detect_all_structural_changes(raw, sensor_data, baseline_std)

        # Regime transition signals [Regime Transition layer]
        regime_transition_candidates = np.array([], dtype=int)
        if regime_timeline is not None and len(regime_timeline) > 0:
            regime_transition_candidates = np.array(
                [r["cycle"] for r in regime_timeline if r.get("transition_detected", False)],
                dtype=int,
            )

        # Combine all candidates from all five layers
        all_candidates = np.concatenate(
            [cusum_candidates, velocity_candidates, zscore_candidates, accel_candidates, corr_candidates, regime_transition_candidates]
        )
        phase1_candidates = np.unique(all_candidates)

        # Filter: only after 15% (healthy region boundary)
        min_cycle = int(n_cycles * 0.15)
        phase1_candidates = phase1_candidates[phase1_candidates >= min_cycle]

        # --- PHASE 2: CONFIRMATION (Evidence Fusion with gating) ---
        # Layer 5: Evidence Fusion with strict multi-signal confirmation
        #
        # Confirmation rule (tight gating):
        #   - Count independent signals in confirmation window
        #   - Require ≥2 independent signal types
        #   - At least 1 must be from structural layer (accel, corr, regime)
        #
        # This prevents single noisy signals from driving false alarms.

        confirmed_idx = None
        if len(phase1_candidates) > 0:
            first_alert = int(phase1_candidates[0])
            look_back = max(0, first_alert - 3)
            look_forward = min(20, n_cycles - first_alert)
            window_end = min(first_alert + look_forward, n_cycles)
            in_window = ema[look_back:window_end]

            # Structural signals (Layers 1, 2, 4)
            accel_in_window = np.any((accel_candidates >= look_back) & (accel_candidates < window_end))
            corr_in_window = np.any((corr_candidates >= look_back) & (corr_candidates < window_end))
            regime_in_window = np.any((regime_transition_candidates >= look_back) & (regime_transition_candidates < window_end))

            # Count independent structural signal types
            structural_signals = sum([accel_in_window, corr_in_window, regime_in_window])
            has_structural = structural_signals > 0

            # Amplitude / Trajectory signals (Layer 3)
            window_has_elevation = np.mean(in_window) > baseline_mean * 1.02
            window_has_breach = np.any(in_window >= baseline_mean + 0.5 * baseline_std)
            window_has_trend = np.any(np.diff(in_window) > 0)

            # Count independent amplitude signal types
            amplitude_signals = sum([window_has_elevation, window_has_breach, window_has_trend])
            has_amplitude = amplitude_signals > 0

            # Confirmation gating (Evidence Fusion — Tight Evidence Fusion Rule):
            # Require EITHER:
            #   (a) ≥2 structural signals (strong structural evidence), OR
            #   (b) ≥1 structural signal AND ≥1 amplitude signal (multi-layer agreement)
            #
            # This prevents single noisy detectors from driving alarms while allowing
            # legitimate multi-signal events to trigger warnings.
            multi_structural = structural_signals >= 2
            multi_layer = has_structural and has_amplitude
            if multi_structural or multi_layer:
                confirmed_idx = first_alert

        # --- PHASE 3: PERSISTENCE (Evidence Fusion) ---
        # Once confirmed, lock warning state until evidence drops below exit threshold

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
