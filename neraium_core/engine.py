from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np

from neraium_core.calibration import DEFAULT_MIN_SAMPLES, QuantileThresholdCalibrator
from neraium_core.metrics import calibrated_components, normalize_components, weighted_composite_score
from neraium_core.windowing import timestamps_from_frames, vector_from_frame, window_from_frames


ALERT_PERSISTENCE_WINDOW = 3
MIN_CONSECUTIVE_WATCH = 2
MIN_CONSECUTIVE_ALERT = 2
CLASSIFICATION_STABILITY_WINDOW = 15


def compute_persistence_features(score_history: deque[float]) -> dict[str, float]:
    values_arr = np.asarray(list(score_history), dtype=float)
    n = int(values_arr.size)
    if n == 0:
        return {
            "history_len": 0.0,
            "rolling_mean": 0.0,
            "rolling_std": 0.0,
            "consecutive_elevated": 0.0,
            "consecutive_high": 0.0,
        }

    window = values_arr[-min(n, 12) :]
    rolling_mean = float(np.mean(window)) if window.size else 0.0
    rolling_std = float(np.std(window)) if window.size else 0.0

    consecutive_elevated = 0
    consecutive_high = 0
    for i in range(n - 1, -1, -1):
        v = float(values_arr[i])
        if v >= 1.5:
            consecutive_elevated += 1
        else:
            break
    for i in range(n - 1, -1, -1):
        v = float(values_arr[i])
        if v >= 2.5:
            consecutive_high += 1
        else:
            break

    return {
        "history_len": float(n),
        "rolling_mean": float(rolling_mean),
        "rolling_std": float(rolling_std),
        "consecutive_elevated": float(consecutive_elevated),
        "consecutive_high": float(consecutive_high),
    }


def classify_drift_state(
    drift_score: float,
    drift_history: deque[float],
    thresholds: tuple[float, float] | None,
) -> str:
    if thresholds is None:
        return "STABLE"

    watch_thr, alert_thr = thresholds
    window = list(drift_history)[-ALERT_PERSISTENCE_WINDOW:]
    consec_watch = 0
    consec_alert = 0
    for v in reversed(window):
        if v > alert_thr:
            consec_alert += 1
            consec_watch += 1
        elif v > watch_thr:
            consec_watch += 1
        else:
            break

    if consec_alert >= MIN_CONSECUTIVE_ALERT:
        return "ALERT"
    if consec_watch >= MIN_CONSECUTIVE_WATCH:
        return "WATCH"
    return "STABLE"


def system_health(drift_score: float, stability_score: float) -> int:
    health = 100.0 - min(float(drift_score) * 20.0, 85.0)
    health += float(stability_score) * 20.0
    return int(round(max(0.0, min(100.0, health))))


def classification_stability(state_history: deque[str]) -> float:
    state_history_list = list(state_history)
    if len(state_history_list) < 2:
        return 1.0
    counts = Counter(state_history_list)
    most_common_count = max(counts.values()) if counts else 0
    return float(most_common_count) / float(len(state_history_list))


def disagreement_factor(components: dict[str, float], tier1_components: set[str]) -> float:
    comp_vals = [float(components.get(k, 0.0)) for k in tier1_components if k in components]
    if not comp_vals:
        return 1.0
    arr_c = np.asarray(comp_vals, dtype=float)
    mean_c = float(np.mean(arr_c))
    std_c = float(np.std(arr_c))
    disagreement = std_c / (mean_c + 1e-6)
    return max(0.7, 1.0 - disagreement * 0.15)


def compute_windows(
    frames: deque[dict[str, Any]],
    *,
    baseline_window: int,
    recent_window: int,
    stride: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[float] | None, list[float] | None]:
    frames_list = list(frames)
    baseline = window_from_frames(
        frames_list,
        size=baseline_window,
        stride=stride,
        recent=False,
    )
    recent = window_from_frames(
        frames_list,
        size=recent_window,
        stride=stride,
        recent=True,
    )
    ts_baseline = timestamps_from_frames(frames_list, size=baseline_window, recent=False)
    ts_recent = timestamps_from_frames(frames_list, size=recent_window, recent=True)
    return baseline, recent, ts_baseline, ts_recent


def append_vector_frame(
    frames: deque[dict[str, Any]],
    frame: dict[str, Any],
    *,
    sensor_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    vector, resolved_order = vector_from_frame(frame, sensor_order=sensor_order or None)
    stored = dict(frame)
    stored["_vector"] = vector
    frames.append(stored)
    return vector, resolved_order


def build_component_confidence(
    *,
    component_keys: list[str],
    evidence_conf: float,
    correlation_ready: bool,
    regime_factor: float,
    tier1_components: set[str],
) -> dict[str, float]:
    component_confidence: dict[str, float] = {k: 0.0 for k in component_keys}
    component_confidence["relational_drift"] = evidence_conf if correlation_ready else 0.0
    component_confidence["spectral"] = evidence_conf if correlation_ready else 0.0
    component_confidence["early_warning"] = evidence_conf
    component_confidence["regime_drift"] = evidence_conf * regime_factor if correlation_ready else 0.0
    for k in list(component_confidence.keys()):
        if k not in tier1_components:
            component_confidence[k] = 0.0
    return component_confidence


def score_with_calibration(
    components: dict[str, float],
    component_confidence: dict[str, float],
) -> tuple[float, dict[str, float], dict[str, float]]:
    canonical = normalize_components(components)
    calibrated = calibrated_components(canonical, component_confidence)
    score = weighted_composite_score(canonical, component_confidence)
    return float(score), canonical, calibrated


def update_calibrator(
    calibrator: QuantileThresholdCalibrator,
    value: float,
) -> tuple[float, float] | None:
    return calibrator.add(float(value))


__all__ = [
    "DEFAULT_MIN_SAMPLES",
    "CLASSIFICATION_STABILITY_WINDOW",
    "append_vector_frame",
    "build_component_confidence",
    "classification_stability",
    "classify_drift_state",
    "compute_persistence_features",
    "compute_windows",
    "disagreement_factor",
    "score_with_calibration",
    "system_health",
    "update_calibrator",
]
