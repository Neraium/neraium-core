"""
Evaluation engine for the FD00x structural early-warning framework.

Responsibilities
----------------
* Dataset loading  – supports all four CMAPSS datasets (FD001–FD004)
* Sensor selection – auto-prunes constant / near-zero-variance channels
* Per-unit metrics – lead time, coverage, false-positive detection
* Aggregate scoring – configurable weighted score for config ranking
* Tuning           – parameter grid sweep on the corrected detector
* Baselines        – three simple reference detectors for comparison
* Result saving    – CSV / JSON output with stable filenames

Degradation proxy
-----------------
CMAPSS training data has no explicit degradation-onset label.  We use:

    degradation_onset = n_cycles − int(n_cycles × degradation_proxy_fraction)

A warning fired before degradation_onset earns positive lead time.
A warning fired in the first ``healthy_fraction × n_cycles`` cycles is a
false positive (alarm in the confirmed-healthy region).

Walk-forward safety
-------------------
All reference statistics (mean, std, covariance, correlation, drift
distribution) are derived *only* from the healthy segment of each unit and
frozen before scoring begins.  No threshold is inferred from future data.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DetectorConfig
from .detector import StructuralDriftDetector, find_warning_index, _apply_ema


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

# CMAPSS column layout (whitespace-delimited, no header):
#   0        unit id
#   1        cycle (time index)
#   2–4      operating settings 1–3
#   5–25     sensor readings s1–s21
_SETTING_COLS = list(range(2, 5))
_SENSOR_COLS = list(range(5, 26))


def load_cmapss_dataset(
    dataset: str,
    train_path: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Load a CMAPSS training file and return a per-unit sensor matrix.

    Parameters
    ----------
    dataset    : "FD001" | "FD002" | "FD003" | "FD004"
    train_path : Explicit file path.  If None, searches for
                 ``train_{dataset}.txt`` in cwd and ``data/`` subdirectory.

    Returns
    -------
    dict mapping unit_id (int) → ndarray of shape (n_cycles, 24)
        Columns:  op_setting_1, op_setting_2, op_setting_3, s1 … s21
        Rows are sorted by cycle (ascending).
    """
    path = _resolve_dataset_path(dataset, train_path)

    rows: Dict[int, List[List[float]]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 26:
                continue
            try:
                unit = int(parts[0])
                cycle = int(parts[1])
                settings = [float(parts[i]) for i in _SETTING_COLS]
                sensors = [float(parts[i]) for i in _SENSOR_COLS]
            except ValueError:
                continue  # skip malformed lines
            row = [float(cycle)] + settings + sensors  # col 0 = cycle for sorting
            rows.setdefault(unit, []).append(row)

    result: Dict[int, np.ndarray] = {}
    for unit, unit_rows in rows.items():
        arr = np.array(sorted(unit_rows, key=lambda r: r[0]))
        # Drop the cycle column; keep settings + sensors (24 cols)
        result[unit] = arr[:, 1:]

    return result


def select_informative_sensors(
    unit_data: Dict[int, np.ndarray],
    config: DetectorConfig,
) -> List[int]:
    """
    Return column indices that carry useful variance for drift detection.

    Strategy:
    - Compute the population std of each column across ALL units.
    - Drop columns whose std < config.min_sensor_variance.
    - If config.exclude_operating_settings, drop the first 3 columns.

    The column layout (after load_cmapss_dataset) is:
        0–2 : op_setting_1–3
        3–23: s1–s21
    """
    # Stack all units to get global variance estimate
    all_data = np.vstack(list(unit_data.values()))
    stds = all_data.std(axis=0)

    cols = list(range(all_data.shape[1]))
    if config.exclude_operating_settings:
        cols = [c for c in cols if c >= 3]

    cols = [c for c in cols if stds[c] >= config.min_sensor_variance]
    return cols


def _resolve_dataset_path(dataset: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"Dataset file not found: {explicit}")
        return explicit

    candidates = [
        f"train_{dataset}.txt",
        os.path.join("data", f"train_{dataset}.txt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    raise FileNotFoundError(
        f"Cannot find training data for {dataset}.  "
        f"Tried: {candidates}.  Pass train_path explicitly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-unit result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class UnitResult:
    """
    Structured per-unit evaluation output.

    All index fields are in units of cycles (0-based).
    """

    unit_id: int
    n_cycles: int

    # Warning
    warning_index: Optional[int]
    threshold_used: float

    # Ground-truth proxies
    healthy_end_index: int       # end of confirmed-healthy region
    degradation_index: int       # estimated degradation onset

    # Derived metrics
    lead_vs_degradation: Optional[int]   # degradation_index − warning_index
    false_positive: bool                  # warning fired in healthy region
    covered: bool                         # warning fired at all

    # Diagnostic fields
    max_ema_before_degradation: float
    baseline_peak: float                  # peak EMA in the healthy region

    # Method tag (for comparison tables)
    method: str = "structural_drift"


def _compute_unit_result(
    unit_id: int,
    scores,               # UnitScores from detector
    config: DetectorConfig,
    method: str = "structural_drift",
) -> UnitResult:
    """Derive all per-unit metrics from a UnitScores object."""
    n = scores.n_cycles
    healthy_end = max(
        config.min_reference_samples,
        int(n * config.healthy_fraction),
    )
    degradation_onset = n - max(1, int(n * config.degradation_proxy_fraction))

    wi = scores.warning_index
    covered = wi is not None
    false_positive = covered and wi < healthy_end

    lead = None
    if wi is not None:
        lead = degradation_onset - wi  # positive = early; negative = late

    ema = scores.ema_drift
    max_ema_before_deg = float(ema[:degradation_onset].max()) if degradation_onset > 0 else 0.0
    baseline_peak = float(ema[:healthy_end].max()) if healthy_end > 0 else 0.0

    return UnitResult(
        unit_id=unit_id,
        n_cycles=n,
        warning_index=wi,
        threshold_used=scores.threshold,
        healthy_end_index=healthy_end,
        degradation_index=degradation_onset,
        lead_vs_degradation=lead,
        false_positive=false_positive,
        covered=covered,
        max_ema_before_degradation=max_ema_before_deg,
        baseline_peak=baseline_peak,
        method=method,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main detector evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_detector(
    config: DetectorConfig,
    unit_data: Dict[int, np.ndarray],
    sensor_cols: List[int],
    override_threshold_std: Optional[float] = None,
    override_persistence: Optional[int] = None,
) -> List[UnitResult]:
    """
    Run the structural drift detector on every unit and collect results.

    Parameters
    ----------
    config               : Run configuration.
    unit_data            : {unit_id: ndarray (n_cycles, n_raw_cols)}.
    sensor_cols          : Column indices to use (from select_informative_sensors).
    override_threshold_std / override_persistence : Override config values.

    Returns
    -------
    List of UnitResult, one per unit.
    """
    detector = StructuralDriftDetector(config)
    results: List[UnitResult] = []

    for unit_id, raw in sorted(unit_data.items()):
        data = raw[:, sensor_cols]
        n = len(data)

        min_len = 2 * config.min_reference_samples
        if n < min_len:
            continue  # skip units too short to evaluate meaningfully

        scores = detector.process_unit(
            data,
            override_threshold_std=override_threshold_std,
            override_persistence=override_persistence,
        )
        results.append(_compute_unit_result(unit_id, scores, config))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate metrics and scoring
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AggregateMetrics:
    """Aggregate evaluation metrics across all units."""

    n_units: int
    coverage: float            # fraction of units where a warning fired
    false_positive_rate: float # fraction of units with FP in healthy region
    mean_lead: float           # mean lead cycles (over units where covered)
    median_lead: float
    min_lead: float
    max_lead: float
    score: float               # composite selection score (configurable)


def compute_aggregate_metrics(
    results: List[UnitResult],
    config: DetectorConfig,
) -> AggregateMetrics:
    """
    Compute aggregate metrics and the composite scoring rule.

    Scoring rule (configurable via config):
        score = mean_lead * lead_w  +  coverage * coverage_w  -  fpr * fpr_penalty

    FPR is penalised ~10× more than coverage is rewarded to ensure that a
    configuration with many false positives cannot out-score a quieter one.
    Modify score_lead_weight / score_coverage_weight / score_fpr_penalty in
    config to change the trade-off.
    """
    if not results:
        return AggregateMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0)

    n = len(results)
    covered = [r for r in results if r.covered]
    fps = [r for r in results if r.false_positive]

    coverage = len(covered) / n
    fpr = len(fps) / n

    leads = [r.lead_vs_degradation for r in covered if r.lead_vs_degradation is not None]
    if leads:
        mean_lead = float(np.mean(leads))
        median_lead = float(np.median(leads))
        min_lead = float(min(leads))
        max_lead = float(max(leads))
    else:
        mean_lead = median_lead = min_lead = max_lead = 0.0

    score = (
        mean_lead * config.score_lead_weight
        + coverage * config.score_coverage_weight
        - fpr * config.score_fpr_penalty
    )

    return AggregateMetrics(
        n_units=n,
        coverage=coverage,
        false_positive_rate=fpr,
        mean_lead=mean_lead,
        median_lead=median_lead,
        min_lead=min_lead,
        max_lead=max_lead,
        score=score,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parameter tuning
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TuningRow:
    threshold_std: float
    persistence: int
    healthy_fraction: float
    coverage: float
    false_positive_rate: float
    mean_lead: float
    score: float


def run_tuning(
    base_config: DetectorConfig,
    unit_data: Dict[int, np.ndarray],
    sensor_cols: List[int],
) -> Tuple[List[TuningRow], TuningRow]:
    """
    Sweep the parameter grid defined in base_config and rank all configurations.

    Tuning is performed on the *corrected* detector (no backdating).
    The base_config is used for all fixed parameters; only threshold_std,
    persistence, and healthy_fraction are swept.

    Returns
    -------
    (all_rows_sorted_desc, best_row)
    """
    rows: List[TuningRow] = []

    grid = list(itertools.product(
        base_config.tune_threshold_values,
        base_config.tune_persistence_values,
        base_config.tune_healthy_fraction_values,
    ))

    for thr, pers, hf in grid:
        cfg = base_config.with_overrides(
            threshold_std=thr,
            persistence=pers,
            healthy_fraction=hf,
        )
        results = evaluate_detector(cfg, unit_data, sensor_cols,
                                    override_threshold_std=thr,
                                    override_persistence=pers)
        metrics = compute_aggregate_metrics(results, cfg)
        rows.append(TuningRow(
            threshold_std=thr,
            persistence=pers,
            healthy_fraction=hf,
            coverage=metrics.coverage,
            false_positive_rate=metrics.false_positive_rate,
            mean_lead=metrics.mean_lead,
            score=metrics.score,
        ))

    rows.sort(key=lambda r: r.score, reverse=True)
    return rows, rows[0] if rows else None


# ─────────────────────────────────────────────────────────────────────────────
# Baseline detectors
# ─────────────────────────────────────────────────────────────────────────────


class BaselineDetector:
    """Abstract base for simple reference detectors."""

    name: str = "baseline"

    def score_unit(
        self,
        data: np.ndarray,
        healthy_end: int,
        config: DetectorConfig,
    ) -> Tuple[np.ndarray, Optional[int], float]:
        """
        Returns (score_series, warning_index, threshold).
        Subclasses must implement this.
        """
        raise NotImplementedError


class RawThresholdBaseline(BaselineDetector):
    """
    Per-channel threshold baseline.

    For each sensor, compute reference mean + k*std from the healthy window.
    Score at each timestep = fraction of sensors exceeding their threshold.
    Warning rule: same persistence logic as the structural detector.

    Deliberately simple: no inter-sensor relationship modelling.
    """

    name = "raw_threshold"

    def score_unit(
        self,
        data: np.ndarray,
        healthy_end: int,
        config: DetectorConfig,
    ) -> Tuple[np.ndarray, Optional[int], float]:
        healthy = data[:healthy_end]
        ref_mean = healthy.mean(axis=0)
        ref_std = healthy.std(axis=0)
        ref_std = np.where(ref_std < 1e-8, 1.0, ref_std)

        thresholds = ref_mean + config.threshold_std * ref_std
        exceed = (data > thresholds).astype(float)
        score = exceed.mean(axis=1)  # fraction of channels exceeded
        ema = _apply_ema(score, config.ema_alpha)

        # Threshold for warning: > 0 means at least one channel exceeded
        warn_thr = 1.0 / max(data.shape[1], 1)
        wi = find_warning_index(ema, warn_thr, config.persistence)
        return ema, wi, warn_thr


class MovingAvgZScoreBaseline(BaselineDetector):
    """
    Moving-average z-score baseline.

    For each sensor, compute the z-score of its moving average relative to
    the reference distribution.  Score = mean z-score across sensors.
    Warning rule: same persistence logic as the structural detector.

    Deliberately simple: no inter-sensor relationship modelling.
    """

    name = "ma_zscore"

    def __init__(self, window: int = 10) -> None:
        self.window = window

    def score_unit(
        self,
        data: np.ndarray,
        healthy_end: int,
        config: DetectorConfig,
    ) -> Tuple[np.ndarray, Optional[int], float]:
        n, d = data.shape
        healthy = data[:healthy_end]
        ref_mean = healthy.mean(axis=0)
        ref_std = healthy.std(axis=0)
        ref_std = np.where(ref_std < 1e-8, 1.0, ref_std)

        w = self.window
        scores = np.zeros(n)
        for i in range(n):
            start = max(0, i - w + 1)
            window_mean = data[start : i + 1].mean(axis=0)
            z = np.abs((window_mean - ref_mean) / ref_std)
            scores[i] = float(z.mean())

        ema = _apply_ema(scores, config.ema_alpha)
        thr = config.threshold_std  # z-score threshold
        wi = find_warning_index(ema, thr, config.persistence)
        return ema, wi, thr


class CUSumBaseline(BaselineDetector):
    """
    Cumulative-sum (CUSUM) drift accumulation baseline.

    Accumulates the mean absolute normalised deviation from the reference
    mean.  Resets when score drops below a small threshold.
    Warning rule: same persistence logic as the structural detector.

    Deliberately simple: no inter-sensor relationship modelling.
    """

    name = "cusum"

    def score_unit(
        self,
        data: np.ndarray,
        healthy_end: int,
        config: DetectorConfig,
    ) -> Tuple[np.ndarray, Optional[int], float]:
        n, d = data.shape
        healthy = data[:healthy_end]
        ref_mean = healthy.mean(axis=0)
        ref_std = healthy.std(axis=0)
        ref_std = np.where(ref_std < 1e-8, 1.0, ref_std)

        normed = np.abs((data - ref_mean) / ref_std)
        deviation = normed.mean(axis=1)

        # CUSUM: accumulate exceedance above ref level (set at 1.0 in normed space)
        ref_level = 1.0
        cusum = np.zeros(n)
        running = 0.0
        for i in range(n):
            running = max(0.0, running + deviation[i] - ref_level)
            cusum[i] = running

        ema = _apply_ema(cusum, config.ema_alpha)
        thr = config.threshold_std * float(np.std(cusum[:healthy_end]) + 1e-6)
        wi = find_warning_index(ema, thr, config.persistence)
        return ema, wi, thr


_BASELINES: List[BaselineDetector] = [
    RawThresholdBaseline(),
    MovingAvgZScoreBaseline(),
    CUSumBaseline(),
]


def compare_baselines(
    config: DetectorConfig,
    unit_data: Dict[int, np.ndarray],
    sensor_cols: List[int],
) -> Dict[str, AggregateMetrics]:
    """
    Evaluate the structural detector and all three baselines on the same data.

    Returns a dict mapping method name → AggregateMetrics.
    All methods use the same persistence / threshold_std / EMA alpha from
    config, and are evaluated with the same FP / coverage / lead definitions.
    """
    results_by_method: Dict[str, List[UnitResult]] = {
        "structural_drift": evaluate_detector(config, unit_data, sensor_cols),
    }

    for baseline in _BASELINES:
        bl_results: List[UnitResult] = []
        for unit_id, raw in sorted(unit_data.items()):
            data = raw[:, sensor_cols]
            n = len(data)
            if n < 2 * config.min_reference_samples:
                continue

            healthy_end = max(
                config.min_reference_samples,
                int(n * config.healthy_fraction),
            )
            degradation_onset = n - max(1, int(n * config.degradation_proxy_fraction))

            ema, wi, thr = baseline.score_unit(data, healthy_end, config)

            covered = wi is not None
            fp = covered and wi < healthy_end
            lead = (degradation_onset - wi) if wi is not None else None

            bl_results.append(UnitResult(
                unit_id=unit_id,
                n_cycles=n,
                warning_index=wi,
                threshold_used=thr,
                healthy_end_index=healthy_end,
                degradation_index=degradation_onset,
                lead_vs_degradation=lead,
                false_positive=fp,
                covered=covered,
                max_ema_before_degradation=float(ema[:degradation_onset].max()),
                baseline_peak=float(ema[:healthy_end].max()),
                method=baseline.name,
            ))
        results_by_method[baseline.name] = bl_results

    return {
        method: compute_aggregate_metrics(res, config)
        for method, res in results_by_method.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_all_datasets(
    base_config: DetectorConfig,
    datasets: Tuple[str, ...],
) -> Dict[str, AggregateMetrics]:
    """
    Run evaluation across multiple CMAPSS datasets and return per-dataset metrics.

    Parameters
    ----------
    base_config : Base configuration (dataset field is overridden per iteration).
    datasets    : Tuple of dataset names e.g. ("FD001", "FD002", "FD004").

    Returns
    -------
    dict mapping dataset name → AggregateMetrics
    """
    summary: Dict[str, AggregateMetrics] = {}

    for ds in datasets:
        cfg = base_config.with_overrides(dataset=ds)
        try:
            unit_data = load_cmapss_dataset(ds, cfg.train_path)
            sensor_cols = select_informative_sensors(unit_data, cfg)
            results = evaluate_detector(cfg, unit_data, sensor_cols)
            summary[ds] = compute_aggregate_metrics(results, cfg)
        except FileNotFoundError as exc:
            print(f"  [skip] {ds}: {exc}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers for unit-level data retrieval (used by plotting)
# ─────────────────────────────────────────────────────────────────────────────


def get_unit_scores_for_plot(
    config: DetectorConfig,
    unit_data: Dict[int, np.ndarray],
    sensor_cols: List[int],
    unit_id: int,
):
    """Return a UnitScores object for a single unit (for diagnostic plots)."""
    detector = StructuralDriftDetector(config)
    data = unit_data[unit_id][:, sensor_cols]
    return detector.process_unit(data)


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_unit_results(results: List[UnitResult], path: str) -> None:
    """Write per-unit results to a CSV file."""
    if not results:
        return
    fieldnames = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def save_tuning_results(rows: List[TuningRow], path: str) -> None:
    """Write tuning grid results to a CSV file."""
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def save_comparison_table(
    comparison: Dict[str, AggregateMetrics],
    path: str,
) -> None:
    """Write baseline comparison to a CSV file."""
    if not comparison:
        return
    fieldnames = ["method"] + list(asdict(next(iter(comparison.values()))).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, metrics in comparison.items():
            row = {"method": method}
            row.update(asdict(metrics))
            writer.writerow(row)


def save_cross_dataset_summary(
    summary: Dict[str, AggregateMetrics],
    path: str,
) -> None:
    """Write cross-dataset summary to a CSV file."""
    if not summary:
        return
    fieldnames = ["dataset"] + list(asdict(next(iter(summary.values()))).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ds, metrics in summary.items():
            row = {"dataset": ds}
            row.update(asdict(metrics))
            writer.writerow(row)


def save_summary_json(metrics: AggregateMetrics, path: str) -> None:
    """Write aggregate metrics to JSON."""
    with open(path, "w") as f:
        json.dump(asdict(metrics), f, indent=2)
