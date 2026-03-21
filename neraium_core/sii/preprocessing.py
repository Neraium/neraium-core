from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataQualitySummary:
    missingness_rate: float
    sensor_coverage: float
    timestamp_irregularity: float
    gate_passed: bool
    statuses: list[str]
    valid_signal_count: int
    total_sensor_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "missingness_rate": round(float(self.missingness_rate), 4),
            "sensor_coverage": round(float(self.sensor_coverage), 4),
            "timestamp_irregularity": round(float(self.timestamp_irregularity), 4),
            "gate_passed": bool(self.gate_passed),
            "statuses": list(self.statuses),
            "valid_signal_count": int(self.valid_signal_count),
            "total_sensor_count": int(self.total_sensor_count),
        }


def clean_frame_values(values: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, value in values.items():
        if value is None:
            out[name] = None
            continue
        try:
            v = float(value)
            if np.isnan(v) or np.isinf(v):
                out[name] = None
            else:
                out[name] = v
        except (TypeError, ValueError):
            out[name] = None
    return out


def compute_data_quality_summary(values: dict[str, float | None]) -> DataQualitySummary:
    total = max(1, len(values))
    valid = 0
    for v in values.values():
        if isinstance(v, (int, float)) and not np.isnan(float(v)) and not np.isinf(float(v)):
            valid += 1
    coverage = float(valid) / float(total)
    missingness = 1.0 - coverage

    statuses: list[str] = []
    if missingness > 0.5:
        statuses.append("DATA_QUALITY_LIMITED")
    if coverage < 0.5:
        statuses.append("LOW_SENSOR_COVERAGE")

    gate_passed = bool(coverage >= 0.5 and valid >= 2)
    return DataQualitySummary(
        missingness_rate=float(max(0.0, min(1.0, missingness))),
        sensor_coverage=float(max(0.0, min(1.0, coverage))),
        timestamp_irregularity=0.0,
        gate_passed=gate_passed,
        statuses=statuses,
        valid_signal_count=int(valid),
        total_sensor_count=int(total),
    )


def data_quality(
    recent_window: np.ndarray,
    *,
    recent_timestamps: list[float] | None,
) -> DataQualitySummary:
    _ = recent_timestamps
    if recent_window.ndim != 2 or recent_window.size == 0:
        return DataQualitySummary(
            missingness_rate=1.0,
            sensor_coverage=0.0,
            timestamp_irregularity=0.0,
            gate_passed=False,
            statuses=["DATA_QUALITY_LIMITED", "LOW_SENSOR_COVERAGE"],
            valid_signal_count=0,
            total_sensor_count=0,
        )

    total = int(recent_window.size)
    miss = int(np.isnan(recent_window).sum())
    missingness_rate = float(miss / max(1, total))
    std_recent = np.nanstd(recent_window, axis=0)
    std_recent = np.nan_to_num(std_recent, nan=0.0, posinf=0.0, neginf=0.0)
    valid_signal_count = int(np.sum(std_recent > 1e-12))
    total_signal_count = int(recent_window.shape[1])
    sensor_coverage = float(valid_signal_count / max(1, total_signal_count))

    statuses: list[str] = []
    if missingness_rate > 0.5:
        statuses.append("DATA_QUALITY_LIMITED")
    if sensor_coverage < 0.5:
        statuses.append("LOW_SENSOR_COVERAGE")
    gate_passed = bool(missingness_rate <= 0.5 and sensor_coverage >= 0.5 and valid_signal_count >= 2)
    return DataQualitySummary(
        missingness_rate=missingness_rate,
        sensor_coverage=sensor_coverage,
        timestamp_irregularity=0.0,
        gate_passed=gate_passed,
        statuses=statuses,
        valid_signal_count=valid_signal_count,
        total_sensor_count=total_signal_count,
    )


def summarize_quality(report: DataQualitySummary) -> dict[str, Any]:
    return report.as_dict()


def impute_column_mean(m: np.ndarray) -> np.ndarray:
    out = np.asarray(m, dtype=float, copy=True)
    if out.size == 0:
        return out
    col_mean = np.nanmean(out, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0, posinf=0.0, neginf=0.0)
    inds = np.where(np.isnan(out))
    if inds[0].size:
        out[inds] = np.take(col_mean, inds[1])
    return out
