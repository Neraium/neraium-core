"""
Core structural drift detector for multivariate sensor time series.

Scientific framing
------------------
This detector monitors how the RELATIONSHIP STRUCTURE between sensors changes
over time relative to a frozen healthy-period reference.  The key insight is
that mechanical / structural degradation typically manifests first as a shift in
inter-sensor dependencies (coupling changes, correlation collapse, covariance
deformation) *before* any individual sensor crosses a raw amplitude threshold.

Algorithm summary
-----------------
1. **Reference fitting (walk-forward safe)**
   From the first ``healthy_fraction × N`` cycles, compute:
   - Reference covariance matrix (normalised sensor space)
   - Reference correlation matrix
   - Empirical drift distribution (used to derive the warning threshold)
   These statistics are FROZEN; they never see future data.

2. **Per-timestep structural drift score**
   For each timestep t, use a rolling window of the most recent
   ``recent_window`` cycles to compute:
   - Covariance shift  = normalised Frobenius distance(recent_cov, ref_cov)
   - Correlation drift = mean |Δ| of upper-triangle entries
   - Mahalanobis drift  = distance(recent_mean, ref_mean | ref_cov_reg^-1)
   - Combined raw drift = weighted sum of all three components

3. **EMA smoothing**
   Mahalanobis drift is EMA-smoothed before fusion; then an additional EMA is
   applied to the fused raw drift to create the warning score:
       ema[t] = alpha × raw[t]  +  (1 − alpha) × ema[t−1]
   This smoothing was preserved from the original fd004 evaluation code.

4. **Persistence-based warning (corrected semantics)**
   A warning fires at the index where ``persistence`` CONSECUTIVE EMA
   exceedances are confirmed.  This is the confirmation index — NOT backdated
   to the first step in the run.  See ``find_warning_index`` for details and
   ``tests/test_warning_logic.py`` for coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import DetectorConfig


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ReferenceStats:
    """
    Frozen reference statistics derived exclusively from the healthy segment.

    All fields are computed once at fit-time and never modified afterwards,
    guaranteeing that no future data leaks into the detector's baseline.
    """

    mean: np.ndarray    # (n_sensors,)  — reference channel means
    std: np.ndarray     # (n_sensors,)  — reference channel stds (normalisation)
    cov: np.ndarray     # (n_sensors, n_sensors) — reference covariance (normed space)
    corr: np.ndarray    # (n_sensors, n_sensors) — reference correlation
    normed_mean: np.ndarray  # (n_sensors,) — reference mean in normalized space
    cov_inv_reg: np.ndarray  # (n_sensors, n_sensors) — pinv(Sigma_ref_reg)
    drift_mean: float   # mean raw drift within the reference window
    drift_std: float    # std  raw drift within the reference window
    n_samples: int      # number of cycles used to build this reference


@dataclass
class UnitScores:
    """
    Per-unit score series produced by one detector run.

    Attributes
    ----------
    raw_drift       : structural drift score at each timestep (before EMA).
    ema_drift       : EMA-smoothed anomaly score (used for warning logic).
    threshold       : absolute EMA threshold used for this unit.
    warning_index   : confirmed warning timestep, or None if never triggered.
    n_cycles        : total number of cycles in the evaluated series.
    reference_stats : the frozen reference used for this unit.
    """

    raw_drift: np.ndarray
    ema_drift: np.ndarray
    threshold: float
    warning_index: Optional[int]
    n_cycles: int
    reference_stats: ReferenceStats


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────


class StructuralDriftDetector:
    """
    Detects structural (relationship-based) drift in multivariate sensor data.

    Parameters
    ----------
    config : DetectorConfig
        All algorithmic parameters.  See ``fd00x.config`` for full docs.

    Usage
    -----
    Typical one-shot call::

        detector = StructuralDriftDetector(config)
        scores = detector.process_unit(sensor_matrix)   # (n_cycles, n_sensors)
        print(scores.warning_index)

    Or split fit / score for fine-grained control::

        ref = detector.fit_reference(healthy_data)
        scores = detector.score_unit(full_data, ref)
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_reference(self, healthy_data: np.ndarray) -> ReferenceStats:
        """
        Compute frozen reference statistics from the healthy segment only.

        This is the walk-forward safety boundary.  ``healthy_data`` must
        contain ONLY the early portion of the unit's life.  The returned
        ``ReferenceStats`` object is immutable and is the sole input for all
        subsequent scoring calls on this unit.

        Parameters
        ----------
        healthy_data : ndarray, shape (n_samples, n_sensors)
            The healthy portion of one unit's sensor readings.

        Returns
        -------
        ReferenceStats
            Frozen reference that can be passed to ``score_unit``.
        """
        if healthy_data.ndim != 2:
            raise ValueError("healthy_data must be 2-D (time × sensors)")
        n, d = healthy_data.shape

        mean = healthy_data.mean(axis=0)
        std = healthy_data.std(axis=0)
        # Protect against constant channels
        std = np.where(std < 1e-8, 1.0, std)

        normed = (healthy_data - mean) / std

        cov = _safe_cov(normed)
        corr = _safe_corr(normed)
        normed_mean = normed.mean(axis=0)
        cov_inv_reg = _compute_regularized_cov_inverse(
            cov,
            regularization=self.config.mahal_regularization,
        )

        # Empirical drift within the reference window — used to set threshold
        ref_drifts = self._compute_raw_drift_series(
            normed=normed,
            ref_cov=cov,
            ref_corr=corr,
            ref_normed_mean=normed_mean,
            ref_cov_inv_reg=cov_inv_reg,
        )[self.config.recent_window :]

        if len(ref_drifts) >= 2:
            drift_mean = float(np.mean(ref_drifts))
            drift_std = float(np.std(ref_drifts))
        else:
            drift_mean = 0.0
            drift_std = 1.0

        return ReferenceStats(
            mean=mean,
            std=std,
            cov=cov,
            corr=corr,
            normed_mean=normed_mean,
            cov_inv_reg=cov_inv_reg,
            drift_mean=drift_mean,
            drift_std=max(drift_std, 1e-6),
            n_samples=n,
        )

    def score_unit(
        self,
        data: np.ndarray,
        ref: ReferenceStats,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        """
        Score a complete unit time series and detect the warning index.

        Parameters
        ----------
        data                  : ndarray (n_cycles, n_sensors)
        ref                   : Frozen reference from ``fit_reference()``.
        override_threshold_std: Override config.threshold_std for this call.
        override_persistence  : Override config.persistence for this call.

        Returns
        -------
        UnitScores
        """
        threshold_std = (
            override_threshold_std
            if override_threshold_std is not None
            else self.config.threshold_std
        )
        persistence = (
            override_persistence
            if override_persistence is not None
            else self.config.persistence
        )

        n, _ = data.shape

        # Normalise using REFERENCE statistics — never using the test segment
        normed = (data - ref.mean) / ref.std

        raw_drift = self._compute_raw_drift_series(
            normed=normed,
            ref_cov=ref.cov,
            ref_corr=ref.corr,
            ref_normed_mean=ref.normed_mean,
            ref_cov_inv_reg=ref.cov_inv_reg,
        )

        # EMA-smoothed anomaly score (preserves EMA logic from original code)
        ema_drift = _apply_ema(raw_drift, self.config.ema_alpha)

        # Threshold must be calibrated in the same score space used for warning
        # detection (EMA drift), computed from healthy EMA only.
        healthy_ema = ema_drift[: ref.n_samples]
        abs_threshold = _compute_ema_threshold(
            healthy_ema=healthy_ema,
            threshold_mode=self.config.threshold_mode,
            threshold_std=threshold_std,
            threshold_percentile=self.config.threshold_percentile,
        )

        # Warning with corrected persistence semantics + optional quality gates
        warning_index = find_warning_index(
            ema_drift,
            abs_threshold,
            persistence,
            require_upward_trend=self.config.require_upward_ema_trend,
            slope_window=self.config.slope_window,
            min_slope=self.config.min_slope,
        )

        return UnitScores(
            raw_drift=raw_drift,
            ema_drift=ema_drift,
            threshold=abs_threshold,
            warning_index=warning_index,
            n_cycles=n,
            reference_stats=ref,
        )

    def process_unit(
        self,
        data: np.ndarray,
        override_threshold_std: Optional[float] = None,
        override_persistence: Optional[int] = None,
    ) -> UnitScores:
        """
        Convenience wrapper: fit reference then score, in one call.

        Splits ``data`` at ``healthy_fraction × n_cycles`` to create the
        reference window.  The split index is always at least
        ``min_reference_samples`` cycles.

        This is the standard entry point for unit evaluation.
        """
        n = len(data)
        healthy_end = max(
            self.config.min_reference_samples,
            int(n * self.config.healthy_fraction),
        )
        healthy_end = min(healthy_end, n - 1)

        ref = self.fit_reference(data[:healthy_end])
        return self.score_unit(
            data,
            ref,
            override_threshold_std=override_threshold_std,
            override_persistence=override_persistence,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _raw_drift_score(
        self,
        window: np.ndarray,
        ref_cov: np.ndarray,
        ref_corr: np.ndarray,
        ref_normed_mean: np.ndarray,
        ref_cov_inv_reg: np.ndarray,
        mahal_ema_prev: float,
    ) -> float:
        """
        Compute raw structural drift between a recent window and the reference.

        The score is a weighted sum of two components that both capture
        RELATIONSHIP CHANGES between sensors (not individual sensor amplitude):

        1. **Covariance shift** — normalised Frobenius distance between the
           window covariance matrix and the reference covariance matrix.
           A large covariance shift indicates that the joint variability
           structure of the sensor group has changed.

        2. **Correlation drift** — mean absolute difference of upper-triangle
           correlation matrix entries.  Correlation drift is sensitive to
           changes in directional coupling between sensors while being robust
           to variance scaling.

        3. **Mahalanobis drift** — distance between recent-window mean and the
           frozen healthy normalized-mean under the regularized inverse
           reference covariance geometry.

        Mahalanobis is smoothed with EMA before fusion to suppress noisy
        pointwise spikes.
        """
        n, d = window.shape
        if n < 2:
            return 0.0

        w_cov = _safe_cov(window)

        # Covariance shift: normalised Frobenius distance
        cov_shift = float(np.linalg.norm(w_cov - ref_cov, "fro")) / (d + 1e-9)

        # Correlation drift: mean |Δ| of upper-triangle entries
        if d > 1:
            w_corr = _safe_corr(window)
            idx = np.triu_indices(d, k=1)
            corr_drift = float(np.mean(np.abs(w_corr[idx] - ref_corr[idx])))
        else:
            corr_drift = 0.0

        mu_recent = window.mean(axis=0)
        delta = mu_recent - ref_normed_mean
        mahal_sq = float(delta @ ref_cov_inv_reg @ delta.T)
        mahal_sq = max(mahal_sq, 0.0)
        mahal_raw = float(np.sqrt(mahal_sq))

        mahal_alpha = (
            self.config.mahal_ema_alpha
            if self.config.mahal_ema_alpha is not None
            else self.config.ema_alpha
        )
        mahal_smooth = (
            mahal_alpha * mahal_raw + (1.0 - mahal_alpha) * mahal_ema_prev
        )

        return (
            self.config.cov_weight * cov_shift
            + self.config.corr_weight * corr_drift
            + self.config.mahal_weight * mahal_smooth
        )

    def _compute_raw_drift_series(
        self,
        normed: np.ndarray,
        ref_cov: np.ndarray,
        ref_corr: np.ndarray,
        ref_normed_mean: np.ndarray,
        ref_cov_inv_reg: np.ndarray,
    ) -> np.ndarray:
        """Compute full raw-drift series with sequential Mahalanobis smoothing."""
        n = len(normed)
        w = self.config.recent_window
        raw_drift = np.zeros(n)
        mahal_ema = 0.0
        for i in range(n):
            start = max(0, i - w)
            window = normed[start : i + 1]
            raw = self._raw_drift_score(
                window=window,
                ref_cov=ref_cov,
                ref_corr=ref_corr,
                ref_normed_mean=ref_normed_mean,
                ref_cov_inv_reg=ref_cov_inv_reg,
                mahal_ema_prev=mahal_ema,
            )
            raw_drift[i] = raw

            if len(window) >= 2:
                mu_recent = window.mean(axis=0)
                delta = mu_recent - ref_normed_mean
                mahal_sq = float(delta @ ref_cov_inv_reg @ delta.T)
                mahal_sq = max(mahal_sq, 0.0)
                mahal_raw = float(np.sqrt(mahal_sq))
                mahal_alpha = (
                    self.config.mahal_ema_alpha
                    if self.config.mahal_ema_alpha is not None
                    else self.config.ema_alpha
                )
                mahal_ema = mahal_alpha * mahal_raw + (1.0 - mahal_alpha) * mahal_ema
        return raw_drift


# ─────────────────────────────────────────────────────────────────────────────
# Warning logic — isolated and unit-testable
# ─────────────────────────────────────────────────────────────────────────────


def find_warning_index(
    scores: np.ndarray,
    threshold: float,
    persistence: int,
    require_upward_trend: bool = False,
    slope_window: int = 3,
    min_slope: float = 0.0,
) -> Optional[int]:
    """
    Find the warning CONFIRMATION index using the corrected persistence rule.

    Definition (corrected)
    ----------------------
    A warning is issued at the index where *persistence* CONSECUTIVE
    exceedances are confirmed.  That is, the returned index is the index of
    the **last step in the confirming run** — NOT the first step.

    This distinction matters for evaluation: backdating to the first step in
    the run would artificially inflate lead-time and undercount late warnings.

    Parameters
    ----------
    scores      : 1-D array of anomaly scores (e.g., EMA drift).
    threshold   : Exceedance threshold (absolute value).
    persistence : Number of consecutive exceedances required (>= 1).
    require_upward_trend : If True, only rising EMA steps count.
    slope_window : Trailing window used to measure sustained EMA momentum.
    min_slope    : Minimum sustained slope required for a step to count.

    Returns
    -------
    int or None
        Index where the warning is confirmed, or None if persistence is never
        satisfied.

    Raises
    ------
    ValueError
        If ``persistence < 1``.

    Examples
    --------
    Simple run of exactly 3:

    >>> import numpy as np
    >>> find_warning_index(np.array([0.1, 0.9, 0.9, 0.9]), 0.5, 3)
    3

    Interrupted run — warning delayed until second run:

    >>> find_warning_index(np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.9]), 0.5, 3)
    5

    Run of 2 when persistence=3 — never triggers:

    >>> find_warning_index(np.array([0.1, 0.9, 0.9, 0.1]), 0.5, 3) is None
    True

    Confirmation index ≠ first-in-run (no backdating):

    >>> find_warning_index(np.array([0.1, 0.9, 0.9, 0.9, 0.9]), 0.5, 3)
    3
    """
    if persistence < 1:
        raise ValueError(f"persistence must be >= 1, got {persistence}")
    if slope_window < 1:
        raise ValueError(f"slope_window must be >= 1, got {slope_window}")

    consecutive = 0
    for i, s in enumerate(scores):
        value = float(s)
        instant_slope = value - float(scores[i - 1]) if i > 0 else 0.0
        upward_ok = (not require_upward_trend) or (i > 0 and instant_slope > 0.0)

        # Sustained slope gate to suppress short-lived EMA spikes and weak
        # noisy excursions while preserving true rising degradation signatures.
        if min_slope <= 0.0:
            # Explicit bypass: non-positive min_slope disables sustained slope gating.
            strong_trend = True
        else:
            if i >= slope_window:
                recent_slope = value - float(scores[i - slope_window])
            else:
                recent_slope = 0.0
            strong_trend = recent_slope > min_slope

        if value >= threshold:
            if consecutive == 0:
                # Gate the START of anomaly buildup by trend quality checks.
                if upward_ok and strong_trend:
                    consecutive = 1
            else:
                # Once above threshold and active, allow plateaus/slight wobble
                # without destroying persistence.
                consecutive += 1

            if consecutive >= persistence:
                # Return the confirmation index — NOT backdated to run start
                return i
        else:
            consecutive = 0

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_cov(data: np.ndarray) -> np.ndarray:
    """Compute covariance matrix; handle 1-D and near-singular cases."""
    n, d = data.shape
    if d == 1:
        return np.array([[float(np.var(data[:, 0]))]])
    return np.cov(data.T)


def _safe_corr(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix; replace NaN with identity entries.

    Near-constant windows (e.g. at series start) produce NaN in numpy's
    corrcoef due to zero-std division.  We suppress that warning and replace
    the NaN positions with identity values (diagonal=1, off-diagonal=0),
    which is the neutral "no relationship measured" value.
    """
    n, d = data.shape
    if d == 1:
        return np.array([[1.0]])
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(data.T)
    nan_mask = np.isnan(corr)
    if nan_mask.any():
        corr = np.where(nan_mask, np.eye(d), corr)
    return corr


def _compute_regularized_cov_inverse(
    cov: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """
    Compute a numerically stable inverse for regularized covariance.

    Uses pseudo-inverse to stay robust for near-singular covariance structure.
    """
    d = cov.shape[0]
    reg = max(float(regularization), 0.0)
    cov_reg = cov + reg * np.eye(d)
    return np.linalg.pinv(cov_reg)


def _apply_ema(series: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply exponential moving average.

    Preserves the EMA-based anomaly scoring logic from the original
    Fd004RiskEscalator in fd004_synthetic.py (alpha=0.25 default).

        ema[t] = alpha * series[t]  +  (1 − alpha) * ema[t−1]
    """
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
    """
    Compute warning threshold from healthy EMA reference distribution only.

    This preserves walk-forward safety (healthy segment only) and ensures score
    and threshold are calibrated on the same distribution.
    """
    if healthy_ema.size == 0:
        return 1e-6

    mode = str(threshold_mode).strip().lower()

    if mode == "mean_std":
        mean = float(np.mean(healthy_ema))
        std = float(np.std(healthy_ema))
        return mean + threshold_std * max(std, 1e-6)

    filtered = _filter_informative_healthy_ema(healthy_ema)
    base = filtered if filtered.size > 0 else healthy_ema

    if mode == "percentile":
        pct = float(np.clip(threshold_percentile, 0.0, 100.0))
        return float(np.percentile(base, pct))

    if mode == "robust_mad":
        median = float(np.median(base))
        mad = float(np.median(np.abs(base - median)))
        return median + threshold_std * max(mad, 1e-6)

    raise ValueError(
        f"Unsupported threshold_mode '{threshold_mode}'. "
        "Expected one of: mean_std, percentile, robust_mad."
    )


def _filter_informative_healthy_ema(healthy_ema: np.ndarray) -> np.ndarray:
    """
    Remove non-informative early EMA values for percentile/MAD thresholding.

    At startup, EMA can include repeated near-zero values caused by short
    partial windows and cold-start behavior. We remove non-finite and near-zero
    entries so robust/percentile calibrations focus on informative healthy
    dynamics.
    """
    finite = healthy_ema[np.isfinite(healthy_ema)]
    if finite.size == 0:
        return finite
    eps = 1e-12
    informative = finite[np.abs(finite) > eps]
    return informative
