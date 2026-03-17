from __future__ import annotations

from typing import Iterable

import numpy as np


def instability_trend(history: Iterable[float]) -> float:
    values = np.asarray(list(history), dtype=float)
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def smooth_drift_velocity(history: Iterable[float], window: int = 3) -> float:
    values = np.asarray(list(history), dtype=float)
    if values.size < 2:
        return 0.0
    diffs = np.diff(values)
    if window <= 1:
        return float(diffs[-1])
    span = min(window, diffs.size)
    return float(np.mean(diffs[-span:]))


def time_to_instability(history: Iterable[float], threshold: float) -> float | None:
    values = np.asarray(list(history), dtype=float)
    if values.size == 0:
        return None

    velocity = smooth_drift_velocity(values)
    if velocity <= 0:
        return None

    remaining = threshold - values[-1]
    if remaining <= 0:
        return 0.0

    return float(remaining / velocity)
