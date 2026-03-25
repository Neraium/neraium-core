from __future__ import annotations

from typing import Sequence

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_dt(timestamps: Sequence[float]) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=float)
    if ts.size < 2:
        return np.asarray([], dtype=float)
    dt = np.diff(ts)
    return np.maximum(1e-6, dt)


def _rolling_std(values: np.ndarray, window: int = 4) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    w = max(2, int(window))
    for i in range(values.size):
        lo = max(0, i - w + 1)
        out[i] = float(np.std(values[lo : i + 1]))
    return out


def derive_temporal_rate_features(
    *,
    recent_window: np.ndarray,
    timestamps: Sequence[float] | None,
) -> dict[str, object]:
    matrix = np.asarray(recent_window, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 1:
        return {"available": False, "reason": "insufficient_window"}

    ts = [float(v) for v in (timestamps if timestamps is not None else [])]
    if len(ts) != matrix.shape[0]:
        ts = list(range(matrix.shape[0]))

    dt = _safe_dt(ts)
    if dt.size < 2:
        return {"available": False, "reason": "insufficient_timestamps"}

    d1 = np.diff(matrix, axis=0)
    rates = d1 / dt[:, None]
    rate_mag = np.linalg.norm(rates, axis=1)

    accel = np.diff(rates, axis=0) / np.maximum(1e-6, dt[1:, None])
    accel_mag = np.linalg.norm(accel, axis=1) if accel.size else np.asarray([], dtype=float)

    pressure_velocity = float(np.quantile(np.abs(rates), 0.8))
    drift_velocity = float(np.mean(rate_mag))
    drift_acceleration = float(np.mean(np.abs(accel_mag))) if accel_mag.size else 0.0

    rolling_dt_stability = float(np.mean(1.0 / (1.0 + _rolling_std(dt) / (np.mean(dt) + 1e-9))))
    gap_irregularity = float(np.std(dt) / (np.mean(dt) + 1e-9))

    shock_threshold = np.mean(rate_mag) + 1.5 * np.std(rate_mag)
    shock_count = int(np.sum(rate_mag > shock_threshold))
    shock_density = float(shock_count / max(1e-9, np.sum(dt)))

    spacing_irregularity = float(np.mean(np.abs(np.diff(dt))) / (np.mean(dt) + 1e-9)) if dt.size >= 2 else 0.0

    compression = float(np.mean(dt) / (np.max(dt) + 1e-9))
    expansion = float(np.min(dt) / (np.mean(dt) + 1e-9))

    return {
        "available": True,
        "delta_t": [round(float(v), 6) for v in dt.tolist()],
        "rolling_delta_t_stability": round(float(_clamp01(rolling_dt_stability)), 6),
        "dx_dt_mean_abs": [round(float(v), 6) for v in np.mean(np.abs(rates), axis=0).tolist()],
        "drift_velocity": round(float(drift_velocity), 6),
        "drift_acceleration": round(float(drift_acceleration), 6),
        "pressure_velocity": round(float(pressure_velocity), 6),
        "shock_density_per_time": round(float(shock_density), 6),
        "event_spacing_irregularity": round(float(_clamp01(spacing_irregularity)), 6),
        "temporal_compression_index": round(float(_clamp01(compression)), 6),
        "temporal_expansion_index": round(float(_clamp01(expansion)), 6),
        "timing_intensity": round(float(_clamp01((drift_velocity + pressure_velocity) / 2.0)), 6),
        "timing_instability": round(float(_clamp01(gap_irregularity + spacing_irregularity)), 6),
    }
