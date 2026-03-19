"""
Data quality gating before analytics.
Prevents poor data quality from masquerading as structural instability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DataQualityReport:
    """Result of data quality checks on baseline and recent windows."""

    missingness_rate: float = 0.0
    stale_sensors: List[str] = field(default_factory=list)
    flatlined_sensors: List[str] = field(default_factory=list)
    timestamp_irregularity: float = 0.0
    sensor_churn: float = 0.0
    outlier_burst_density: float = 0.0
    statuses: List[str] = field(default_factory=list)
    sensor_coverage: float = 1.0
    variability_coverage: float = 1.0
    valid_signal_count: int = 0
    total_sensors: int = 0
    gate_passed: bool = True

    def to_dict(self) -> dict:
        return {
            "missingness_rate": self.missingness_rate,
            "stale_sensors": self.stale_sensors,
            "flatlined_sensors": self.flatlined_sensors,
            "timestamp_irregularity": self.timestamp_irregularity,
            "sensor_churn": self.sensor_churn,
            "outlier_burst_density": self.outlier_burst_density,
            "statuses": self.statuses,
            "sensor_coverage": self.sensor_coverage,
            "variability_coverage": self.variability_coverage,
            "valid_signal_count": self.valid_signal_count,
            "total_sensors": self.total_sensors,
            "gate_passed": self.gate_passed,
        }


# Status constants for downstream gating
DATA_QUALITY_LIMITED = "DATA_QUALITY_LIMITED"
INSUFFICIENT_VARIABILITY = "INSUFFICIENT_VARIABILITY"
LOW_SENSOR_COVERAGE = "LOW_SENSOR_COVERAGE"
TIMESTAMP_IRREGULAR = "TIMESTAMP_IRREGULAR"
HIGH_SENSOR_CHURN = "HIGH_SENSOR_CHURN"

# Default thresholds (configurable via caller)
DEFAULT_MIN_SENSORS = 2
DEFAULT_MAX_MISSINGNESS = 0.5
DEFAULT_MIN_VARIABILITY_COVERAGE = 0.25
DEFAULT_FLATLINE_STD_THRESHOLD = 1e-12
DEFAULT_STALE_NAN_RATE = 0.8
DEFAULT_MAX_CHURN = 0.8
DEFAULT_MAX_IRREGULARITY = 0.5


def compute_data_quality(
    baseline_matrix: np.ndarray,
    recent_matrix: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    timestamps_baseline: Optional[List[float]] = None,
    timestamps_recent: Optional[List[float]] = None,
    *,
    min_sensors: int = DEFAULT_MIN_SENSORS,
    max_missingness: float = DEFAULT_MAX_MISSINGNESS,
    min_variability_coverage: float = DEFAULT_MIN_VARIABILITY_COVERAGE,
    flatline_std_threshold: float = DEFAULT_FLATLINE_STD_THRESHOLD,
    stale_nan_rate: float = DEFAULT_STALE_NAN_RATE,
    max_churn: float = DEFAULT_MAX_CHURN,
    max_irregularity: float = DEFAULT_MAX_IRREGULARITY,
) -> DataQualityReport:
    """
    Compute data quality metrics and statuses.
    Returns a report; gate_passed is False if quality is too poor for production analytics.
    """
    baseline = np.asarray(baseline_matrix, dtype=float)
    recent = np.asarray(recent_matrix, dtype=float)
    if baseline.ndim != 2 or recent.ndim != 2:
        return DataQualityReport(
            statuses=[DATA_QUALITY_LIMITED],
            gate_passed=False,
        )
    n_sensors = baseline.shape[1]
    if sensor_names is None:
        sensor_names = [f"s_{i}" for i in range(n_sensors)]
    else:
        sensor_names = list(sensor_names)[:n_sensors]

    report = DataQualityReport(total_sensors=n_sensors)

    # Missingness per sensor (recent window)
    nan_count = np.sum(np.isnan(recent), axis=0)
    total_obs = recent.shape[0] * recent.shape[1]
    report.missingness_rate = float(np.sum(nan_count) / total_obs) if total_obs else 0.0

    # Stale sensors: mostly NaN in recent
    obs_per_sensor = recent.shape[0]
    stale = (nan_count / max(1, obs_per_sensor)) >= stale_nan_rate
    report.stale_sensors = [sensor_names[i] for i in range(n_sensors) if i < len(sensor_names) and stale[i]]

    # Flatlined: near-zero std in recent (using nanstd)
    safe_recent = np.nan_to_num(recent, nan=0.0)
    std_recent = np.nanstd(recent, axis=0)
    std_recent = np.nan_to_num(std_recent, nan=0.0)
    flatlined = std_recent <= flatline_std_threshold
    report.flatlined_sensors = [sensor_names[i] for i in range(n_sensors) if i < len(sensor_names) and flatlined[i]]

    # Valid mask: nonzero variance in recent or baseline
    std_baseline = np.nanstd(baseline, axis=0)
    std_baseline = np.nan_to_num(std_baseline, nan=0.0)
    valid_mask = (std_recent > flatline_std_threshold) | (std_baseline > flatline_std_threshold)
    report.valid_signal_count = int(np.sum(valid_mask))

    # Variability coverage: fraction of sensors with usable variance
    report.variability_coverage = report.valid_signal_count / max(1, n_sensors)
    if report.variability_coverage < min_variability_coverage:
        report.statuses.append(INSUFFICIENT_VARIABILITY)
    if report.valid_signal_count < min_sensors:
        report.statuses.append(LOW_SENSOR_COVERAGE)
    report.sensor_coverage = 1.0 - (len(report.stale_sensors) / max(1, n_sensors))
    if report.sensor_coverage < 0.5:
        report.statuses.append(LOW_SENSOR_COVERAGE)
    if report.missingness_rate > max_missingness:
        report.statuses.append(DATA_QUALITY_LIMITED)

    # Timestamp irregularity (if timestamps provided)
    if timestamps_baseline is not None and len(timestamps_baseline) >= 2:
        ts = np.array(timestamps_baseline, dtype=float)
        gaps = np.diff(ts)
        if np.all(gaps > 0):
            cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))
            report.timestamp_irregularity = min(1.0, cv)
            if report.timestamp_irregularity > max_irregularity:
                report.statuses.append(TIMESTAMP_IRREGULAR)
    if timestamps_recent is not None and len(timestamps_recent) >= 2 and TIMESTAMP_IRREGULAR not in report.statuses:
        ts = np.array(timestamps_recent, dtype=float)
        gaps = np.diff(ts)
        if np.all(gaps > 0):
            cv = float(np.std(gaps) / (np.mean(gaps) + 1e-12))
            report.timestamp_irregularity = max(report.timestamp_irregularity, min(1.0, cv))
            if report.timestamp_irregularity > max_irregularity:
                report.statuses.append(TIMESTAMP_IRREGULAR)

    # Sensor churn: Jaccard distance of "valid" sensors between baseline and recent
    valid_baseline = (std_baseline > flatline_std_threshold).astype(int)
    valid_recent = (std_recent > flatline_std_threshold).astype(int)
    intersection = np.sum(valid_baseline & valid_recent)
    union = np.sum(valid_baseline | valid_recent)
    report.sensor_churn = 1.0 - (intersection / union) if union else 0.0
    if report.sensor_churn > max_churn:
        report.statuses.append(HIGH_SENSOR_CHURN)

    # Outlier burst density: proportion of consecutive runs of robust z > 2
    if safe_recent.size >= 3:
        med = np.median(safe_recent)
        mad = np.median(np.abs(safe_recent - med)) + 1e-12
        z = np.abs((safe_recent - med) / mad)
        over = (z > 2.0).astype(float)
        runs = np.diff(over, axis=0)
        burst_starts = np.sum(runs == 1) if runs.size else 0
        report.outlier_burst_density = burst_starts / max(1, recent.shape[0] * recent.shape[1])
    else:
        report.outlier_burst_density = 0.0

    if not report.statuses:
        report.statuses = []
    report.gate_passed = (
        report.valid_signal_count >= min_sensors
        and report.missingness_rate <= max_missingness
        and report.variability_coverage >= min_variability_coverage
    )
    return report
