from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def instability_trend_regression(series: Sequence[float]) -> float:
    """Linear trend estimate for instability series."""
    y = np.asarray(list(series), dtype=float)

    if len(y) < 5:
        return 0.0

    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def time_to_instability_regression(
    series: Sequence[float],
    threshold: float = 1.5,
) -> float | None:
    """Project time to threshold crossing from linear regression."""
    y = np.asarray(list(series), dtype=float)

    if len(y) < 5:
        return None

    x = np.arange(len(y), dtype=float)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]

    current = float(y[-1])
    if slope <= 0:
        return None

    tti = (float(threshold) - current) / float(slope)
    return float(max(0.0, tti))


# Backward-compatible wrappers
def instability_trend(series: Sequence[float]) -> float:
    return instability_trend_regression(series)


def time_to_instability(series: Sequence[float], threshold: float = 1.5) -> float | None:
    return time_to_instability_regression(series, threshold=threshold)