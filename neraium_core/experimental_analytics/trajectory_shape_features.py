from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _oscillation_index(series: np.ndarray) -> float:
    if series.size < 4:
        return 0.0
    diffs = np.diff(series)
    sign_changes = 0
    prev = 0
    for d in diffs:
        s = 1 if d > 1e-9 else (-1 if d < -1e-9 else 0)
        if s == 0:
            continue
        if prev != 0 and s != prev:
            sign_changes += 1
        prev = s
    denom = max(1, len(diffs) - 1)
    return _clamp01(sign_changes / denom)


def derive_trajectory_shape_features(
    *,
    recent_window: Iterable[Sequence[float]],
    short_horizon: int = 4,
    mid_horizon: int = 8,
) -> dict[str, object]:
    matrix = np.asarray(list(recent_window), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 4 or matrix.shape[1] < 1:
        return {"available": False, "reason": "insufficient_window"}

    center = np.mean(matrix, axis=0, keepdims=True)
    spread = np.std(matrix, axis=0, keepdims=True) + 1e-9
    normalized = (matrix - center) / spread

    aggregate = np.linalg.norm(normalized - normalized[0:1, :], axis=1)
    diffs = np.diff(aggregate)
    second = np.diff(aggregate, n=2) if aggregate.size >= 3 else np.asarray([], dtype=float)

    oscillation = _oscillation_index(aggregate)
    punctuated = _clamp01(float(np.max(np.abs(diffs))) / (float(np.mean(np.abs(diffs))) + 1e-9))
    monotonicity = _clamp01(abs(float(np.mean(np.sign(diffs)))))

    valley_idx = int(np.argmin(aggregate))
    recovery_then_failure = 0.0
    if 0 < valley_idx < len(aggregate) - 1:
        before = float(aggregate[valley_idx] - aggregate[0])
        after = float(aggregate[-1] - aggregate[valley_idx])
        recovery_then_failure = _clamp01(max(0.0, -before) * max(0.0, after))

    per_feature_slopes = np.mean(np.diff(normalized, axis=0), axis=0)
    slope_disagreement = _clamp01(float(np.std(per_feature_slopes)) * 0.9)

    short_tail = normalized[-min(short_horizon, normalized.shape[0]) :]
    mid_tail = normalized[-min(mid_horizon, normalized.shape[0]) :]
    short_embedding = np.mean(np.diff(short_tail, axis=0), axis=0)
    mid_embedding = np.mean(np.diff(mid_tail, axis=0), axis=0)

    return {
        "available": True,
        "recent_slope_pattern": round(float(np.mean(diffs)), 6),
        "recent_curvature_pattern": round(float(np.mean(second)) if second.size else 0.0, 6),
        "rolling_drift_composition": [round(float(v), 6) for v in np.abs(per_feature_slopes)],
        "residual_trend_signature": round(float(np.mean(aggregate[-4:] - np.mean(aggregate[:-1]))), 6),
        "windowed_directional_embedding": {
            "short": [round(float(v), 6) for v in short_embedding],
            "mid": [round(float(v), 6) for v in mid_embedding],
        },
        "shape_scores": {
            "steady_deterioration": round(float(monotonicity * (1.0 - oscillation)), 6),
            "oscillatory_deterioration": round(float(oscillation), 6),
            "punctuated_shift": round(float(_clamp01(punctuated - 0.6)), 6),
            "monotonic_collapse": round(float(monotonicity), 6),
            "recovery_then_failure": round(float(recovery_then_failure), 6),
            "divergent_failure_modes": round(float(slope_disagreement), 6),
        },
    }
