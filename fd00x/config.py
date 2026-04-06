"""
Configuration for the FD00x structural early-warning evaluation framework.

All tunable parameters live here. Use named presets for common trust/sensitivity
trade-offs. The 'default_trusted' preset is the recommended default — it
prioritises trustworthiness over aggressiveness.

Walk-forward safety contract
-----------------------------
`healthy_fraction` determines what share of each unit's life is used as the
FROZEN reference window.  Reference statistics (covariance, correlation, drift
distribution) are derived *only* from that early segment and never updated
using future data.  Threshold estimation is therefore leak-free by construction.

Scoring rule (configurable)
----------------------------
    score = mean_lead * lead_w  +  coverage * coverage_w  -  fpr * fpr_penalty

The FPR term is penalised ~10× more than coverage is rewarded so that a
configuration with many false positives cannot out-score a quieter one.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class DetectorConfig:
    """
    Complete, self-contained configuration for one evaluation run.

    Every knob that affects scientific output is captured here so that a saved
    config.json is sufficient to reproduce any result.
    """

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "FD004"
    """CMAPSS dataset to evaluate: FD001 | FD002 | FD003 | FD004."""

    train_path: Optional[str] = None
    """
    Explicit path to the train_FD00x.txt file.
    If None, the framework looks for train_{dataset}.txt in the working
    directory or a 'data/' subdirectory.
    """

    # ── Sensor selection ─────────────────────────────────────────────────────
    min_sensor_variance: float = 1e-4
    """
    Sensors whose standard deviation across the full dataset is below this
    value are excluded from drift computation.  CMAPSS contains several
    near-constant sensor channels that carry no information.
    """

    exclude_operating_settings: bool = True
    """
    Whether to exclude the three operating-condition columns (op_setting_1–3)
    from structural drift computation.  Operating conditions are external
    inputs, not system health signals.
    """

    # ── Reference window (walk-forward safety) ───────────────────────────────
    healthy_fraction: float = 0.35
    """
    Fraction of each unit's life treated as the healthy reference window.

    Reference covariance and correlation statistics are computed ONLY from
    the first ``healthy_fraction * n_cycles`` steps and then FROZEN.
    No future data ever influences the reference.  Typical range: 0.25–0.45.
    """

    min_reference_samples: int = 20
    """
    Minimum number of cycles required to fit a valid reference.  Units
    shorter than 2 × this value are skipped.
    """

    # ── Rolling window for recent geometry ───────────────────────────────────
    recent_window: int = 15
    """
    Rolling window (in cycles) used to estimate the *recent* covariance and
    correlation at each timestep.  Smaller = more reactive; larger = smoother.
    """

    # ── Structural drift weights ──────────────────────────────────────────────
    cov_weight: float = 0.40
    """
    Weight for the covariance-shift component of the structural drift score.
    Covariance shift = normalised Frobenius distance between recent and
    reference covariance matrices.
    """

    corr_weight: float = 0.25
    """
    Weight for the correlation-drift component.
    Correlation drift = mean absolute difference of upper-triangle entries
    between recent and reference correlation matrices.
    """

    mahal_weight: float = 0.35
    """
    Weight for the Mahalanobis-drift component.
    Mahalanobis drift is computed from the distance between the recent-window
    mean vector and the frozen healthy reference mean/covariance in normalized
    sensor space.
    """

    mahal_regularization: float = 1e-4
    """
    Diagonal regularization added to reference covariance before inversion:

        Sigma_ref_reg = Sigma_ref + mahal_regularization * I

    This keeps Mahalanobis computation numerically stable for near-singular
    covariance structure.
    """

    mahal_ema_alpha: Optional[float] = None
    """
    Optional EMA smoothing factor for Mahalanobis drift BEFORE combination.
    If None, reuse ``ema_alpha``.
    """

    # Together ``cov_weight + corr_weight + mahal_weight`` should equal 1.0
    # for interpretability, but the code does not enforce this.

    # ── EMA smoothing ─────────────────────────────────────────────────────────
    ema_alpha: float = 0.25
    """
    EMA smoothing factor for the raw structural drift score.
    Preserves the EMA-based anomaly scoring from the original fd004 code.

        ema[t] = alpha * raw[t]  +  (1 - alpha) * ema[t-1]

    Lower alpha → smoother, slower to react.
    Higher alpha → noisier, faster to react.
    """

    # ── Warning trigger (corrected persistence rule) ──────────────────────────
    threshold_std: float = 2.5
    """
    Warning threshold expressed in units of reference-window standard
    deviations above the reference-window mean drift.  Higher = more
    conservative (fewer false positives, later warnings).
    """

    persistence: int = 5
    """
    Number of CONSECUTIVE EMA exceedances required to confirm a warning.

    WARNING SEMANTICS (corrected):
        The warning index is the index of the Nth consecutive exceedance —
        the CONFIRMATION step.  It is NOT backdated to the first step in
        the exceedance run.  See ``fd00x.detector.find_warning_index``.
    """

    threshold_mode: str = "mean_std"
    """
    Threshold derivation mode in EMA score space.

    Supported values:
    - ``mean_std``:
        threshold = healthy_ema_mean + threshold_std * healthy_ema_std
    - ``percentile``:
        threshold = percentile(healthy_ema_filtered, threshold_percentile)
    - ``robust_mad``:
        threshold = median(healthy_ema_filtered) + threshold_std * MAD

    ``healthy_ema`` is derived only from the early reference segment, preserving
    walk-forward safety and avoiding score/threshold distribution mismatch.
    """

    threshold_percentile: float = 95.0
    """
    Percentile used when ``threshold_mode == "percentile"``.
    """

    require_upward_ema_trend: bool = True
    """
    If True, an exceedance only counts toward persistence when EMA is rising
    versus the previous step.
    """

    slope_window: int = 3
    """
    Window length used for sustained momentum gating in warning confirmation.

    Recent EMA slope is estimated from this trailing segment.  The gate is
    designed to suppress short-lived spikes and weak noisy excursions while
    preserving warnings for genuine rising degradation signatures.
    """

    min_slope: float = 0.02
    """
    Minimum sustained EMA slope required for a step to count toward
    persistence confirmation.

    Set to 0.0 to disable sustained slope gating.
    """

    # ── Degradation proxy ─────────────────────────────────────────────────────
    degradation_proxy_fraction: float = 0.30
    """
    Fraction of each unit's total life counted as the 'degraded' region,
    measured from the end.  For a unit with N cycles:

        degradation_onset = N - int(N * degradation_proxy_fraction)

    A warning fired before degradation_onset gives positive lead time.
    A warning fired in the first ``healthy_fraction * N`` cycles is a FP.
    This proxy is intentionally simple; replace with RUL labels if available.
    """

    # ── Scoring / model-selection ─────────────────────────────────────────────
    score_lead_weight: float = 1.0
    score_coverage_weight: float = 40.0
    score_fpr_penalty: float = 400.0
    """
    Configurable scoring rule used for config ranking in tuning:

        score = mean_lead * lead_w
              + coverage  * coverage_w
              - fpr       * fpr_penalty

    FPR is penalised ~10× more than coverage is rewarded, so a configuration
    with many false positives cannot beat a quieter one.
    Modify these weights to adjust the trust / sensitivity trade-off.
    """

    # ── Tuning grid ───────────────────────────────────────────────────────────
    tune_threshold_values: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5)
    """threshold_std values swept during tuning."""

    tune_persistence_values: Tuple[int, ...] = (3, 5, 7, 10)
    """persistence values swept during tuning."""

    tune_healthy_fraction_values: Tuple[float, ...] = (0.30, 0.35, 0.40)
    """healthy_fraction values swept during tuning (optional; set to single
    value to hold fixed)."""

    # ── Runtime / output ─────────────────────────────────────────────────────
    mode: str = "evaluate"
    """
    Runtime mode.  One of:
        evaluate            – run the structural detector, save results
        tune                – sweep parameter grid, save tuning_results.csv
        plot                – generate diagnostic plots (requires prior evaluate)
        compare_baselines   – run detector + baselines, save comparison table
        all                 – run evaluate, tune, compare_baselines, and plot
    Default is 'evaluate' (conservative; tuning is never the default).
    """

    random_seed: int = 42
    """Seed for any stochastic components (ensures reproducibility)."""

    output_root: str = "outputs"
    """Root directory.  Each run writes to outputs/<run_name>/."""

    run_name: Optional[str] = None
    """
    Human-readable run label.  Defaults to a timestamp string if None.
    Used as the subdirectory name under output_root.
    """

    # ── Atomic layer detector controls (v2) ─────────────────────────────────
    window_size: int = 120
    """Atomic monitor rolling buffer size for online micro-detectors."""

    forgetting_factor: float = 0.98
    """Exponential forgetting used by online SDE-style updates."""

    sde_dt: float = 1.0
    """Delta-t used by micro-dynamics drift/diffusion tracking."""

    te_k: int = 3
    """Placeholder setting reserved for TE estimator replacement."""

    latent_state_count: int = 4
    """Number of latent regimes used by the v1 occupancy approximation."""

    dmd_rank: int = 4
    """Rank for DMD-like dynamic mode approximation."""

    compute_interval_structure: int = 1
    """Decimation interval for micro-structure updates."""

    compute_interval_state: int = 5
    """Decimation interval for micro-state updates."""

    compute_interval_topology: int = 10
    """Decimation interval for micro-topology updates."""

    compute_interval_events: int = 1
    """Decimation interval for micro-time event updates."""

    green_yellow: float = 0.60
    """Atomic fused score threshold for GREEN→YELLOW transitions."""

    yellow_red: float = 0.80
    """Atomic fused score threshold for YELLOW→RED transitions."""

    detector_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "micro_dynamics": 0.25,
            "micro_structure": 0.20,
            "micro_time": 0.15,
            "micro_state": 0.20,
            "micro_topology": 0.20,
        }
    )
    """Per-layer fusion weights for the atomic anomaly fusion engine."""

    event_level_std: float = 1.0
    """Level crossing z-threshold used by micro-time event extraction."""

    # ── Atomic score stabilization ───────────────────────────────────────────
    score_ema_alpha: float = 0.3
    """EMA smoothing alpha applied to the fused atomic score before alert-state
    transitions.  Lower = smoother / more conservative.  Range: (0, 1]."""

    fusion_activation_floor: float = 0.15
    """Minimum normalized component score to count as 'active' for gating."""

    fusion_min_active: int = 2
    """Minimum number of active components required before producing a high
    fused score.  If fewer are active, the fused score is downweighted."""

    fusion_downweight_factor: float = 0.3
    """Multiplier applied to the fused score when fewer than
    ``fusion_min_active`` components are active."""

    calibration_enabled: bool = True
    """Enable calibrated FP-suppression monitor wrapper (conformal/consensus/BH)."""

    conformal_enabled: bool = True
    conformal_alpha: float = 0.05
    conformal_window: int = 1000
    operational_modes: Optional[int] = None
    maintenance_windows: bool = False
    diurnal_patterns: bool = True
    consensus_required: int = 2
    consensus_window: int = 10
    threshold_adaptation: bool = True
    target_fp_rate: float = 0.01
    fp_history_window: int = 1000
    bh_enabled: bool = True
    bh_fdr_target: float = 0.05
    min_alert_duration: int = 5
    cooldown_period: int = 50

    # ─────────────────────────────────────────────────────────────────────────
    # Serialisation helpers
    # ─────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)

    def save(self, path: str) -> None:
        """Write config to a JSON file for run reproducibility."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        """Reconstruct a DetectorConfig from a plain dict (e.g., loaded JSON)."""
        _TUPLE_FIELDS = {
            "tune_threshold_values",
            "tune_persistence_values",
            "tune_healthy_fraction_values",
        }
        d2 = dict(d)
        for k in _TUPLE_FIELDS:
            if k in d2 and d2[k] is not None:
                d2[k] = tuple(d2[k])
        # Drop unknown keys for forward compatibility
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        d2 = {k: v for k, v in d2.items() if k in known}
        return cls(**d2)

    @classmethod
    def load(cls, path: str) -> "DetectorConfig":
        """Load config from a JSON file written by ``save()``."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def with_overrides(self, **kwargs) -> "DetectorConfig":
        """Return a copy of this config with selected fields overridden."""
        d = self.to_dict()
        d.update(kwargs)
        return self.from_dict(d)


# ─────────────────────────────────────────────────────────────────────────────
# Named presets
# ─────────────────────────────────────────────────────────────────────────────
#
# Choose a preset that matches your trust requirements.
# default_trusted is the runtime default and emphasises trustworthiness.

PRESETS: Dict[str, DetectorConfig] = {
    # default_trusted: conservative preset calibrated for the normalised atomic layer.
    # Uses 99th-percentile thresholding so the threshold is set at the empirical
    # maximum of the healthy-period EMA — healthy warm-up spikes can never persist
    # above this threshold for the required 5 consecutive steps.
    "default_trusted": DetectorConfig(
        threshold_std=1.5,
        threshold_mode="percentile",
        threshold_percentile=99.0,
        persistence=5,
        require_upward_ema_trend=True,
        slope_window=3,
        min_slope=0.02,
        score_fpr_penalty=400.0,
        score_coverage_weight=40.0,
    ),
    "balanced": DetectorConfig(
        threshold_std=1.6,
        threshold_mode="percentile",
        threshold_percentile=97.0,
        persistence=4,
        require_upward_ema_trend=True,
        slope_window=3,
        min_slope=0.025,
        ema_alpha=0.22,
        score_fpr_penalty=400.0,
        score_coverage_weight=40.0,
    ),
    "strict": DetectorConfig(
        threshold_std=1.75,
        threshold_mode="percentile",
        threshold_percentile=99.5,
        persistence=7,
        require_upward_ema_trend=True,
        slope_window=4,
        min_slope=0.03,
        ema_alpha=0.20,
        score_fpr_penalty=450.0,
        score_coverage_weight=35.0,
    ),
}

DEFAULT_PRESET: str = "default_trusted"

ALL_DATASETS: Tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
