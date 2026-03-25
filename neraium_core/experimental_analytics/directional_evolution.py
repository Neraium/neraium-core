from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.zeros_like(vec)
    return vec / norm


def _window_slope(matrix: np.ndarray, horizon: int) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[1], dtype=float)
    tail = matrix[-min(matrix.shape[0], horizon) :]
    x = np.arange(tail.shape[0], dtype=float)
    x_centered = x - float(np.mean(x))
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        return np.zeros(matrix.shape[1], dtype=float)
    centered = tail - np.mean(tail, axis=0, keepdims=True)
    return np.dot(x_centered, centered) / denom


def _group_contributions(values: np.ndarray, group_size: int) -> list[dict[str, object]]:
    if values.size == 0:
        return []
    abs_values = np.abs(values)
    total = float(np.sum(abs_values)) + 1e-12
    groups: list[dict[str, object]] = []
    for start in range(0, int(values.size), max(1, group_size)):
        end = min(int(values.size), start + max(1, group_size))
        score = float(np.sum(abs_values[start:end])) / total
        groups.append({
            "group_id": f"feature_group_{start // max(1, group_size)}",
            "feature_index_range": [start, end - 1],
            "relative_drift_contribution": round(float(score), 4),
        })
    groups.sort(key=lambda item: float(item["relative_drift_contribution"]), reverse=True)
    return groups


def derive_directional_evolution_features(
    *,
    recent_window: Iterable[Sequence[float]],
    feature_names: Sequence[str] | None = None,
    short_horizon: int = 4,
    mid_horizon: int = 8,
    group_size: int = 4,
) -> dict[str, object]:
    matrix = np.asarray(list(recent_window), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[1] < 1:
        return {
            "available": False,
            "reason": "insufficient_window",
            "directional_divergence_score": 0.0,
            "dominant_feature_groups": [],
        }

    diffs = np.diff(matrix, axis=0)
    mean_first_diff = np.mean(diffs, axis=0)
    normalized_diff = _safe_normalize(mean_first_diff)

    short_slope = _window_slope(matrix, short_horizon)
    mid_slope = _window_slope(matrix, mid_horizon)

    drift_signature = np.mean(np.abs(diffs), axis=0)
    drift_total = float(np.sum(drift_signature)) + 1e-12
    relative_contrib = drift_signature / drift_total
    polarity = float(np.sum(np.sign(mean_first_diff) * relative_contrib))

    cosine = float(np.dot(_safe_normalize(short_slope), _safe_normalize(mid_slope)))
    cosine = max(-1.0, min(1.0, cosine))

    curvature = np.zeros(matrix.shape[1], dtype=float)
    if matrix.shape[0] >= 4:
        curvature = np.mean(np.diff(matrix, n=2, axis=0), axis=0)

    directional_divergence = float(
        0.45 * (1.0 - max(0.0, cosine))
        + 0.35 * np.std(_safe_normalize(short_slope))
        + 0.2 * np.std(_safe_normalize(mid_slope))
    )

    names = list(feature_names or [f"feature_{i}" for i in range(matrix.shape[1])])
    per_feature = []
    for idx, name in enumerate(names[: matrix.shape[1]]):
        per_feature.append(
            {
                "feature": name,
                "mean_diff": round(float(mean_first_diff[idx]), 5),
                "short_slope": round(float(short_slope[idx]), 5),
                "mid_slope": round(float(mid_slope[idx]), 5),
                "curvature": round(float(curvature[idx]), 5),
                "relative_drift_contribution": round(float(relative_contrib[idx]), 5),
            }
        )

    dominant = sorted(per_feature, key=lambda item: float(item["relative_drift_contribution"]), reverse=True)[:5]

    return {
        "available": True,
        "normalized_first_difference_vector": [round(float(v), 6) for v in normalized_diff],
        "rolling_slope_vector_short": [round(float(v), 6) for v in short_slope],
        "rolling_slope_vector_mid": [round(float(v), 6) for v in mid_slope],
        "feature_wise_drift_signature": [round(float(v), 6) for v in drift_signature],
        "change_pattern_cosine_direction": round(float(cosine), 6),
        "relative_feature_drift_contribution": [round(float(v), 6) for v in relative_contrib],
        "curvature_profile": [round(float(v), 6) for v in curvature],
        "directional_divergence_score": round(_clamp01(directional_divergence), 6),
        "directional_polarity_score": round(max(-1.0, min(1.0, polarity)), 6),
        "dominant_features": dominant,
        "dominant_feature_groups": _group_contributions(drift_signature, group_size=group_size),
    }
