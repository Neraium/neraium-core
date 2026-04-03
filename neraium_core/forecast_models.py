from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np


def _fit_ar1(series: Sequence[float]) -> tuple[float, float] | None:
    """
    Fit a simple AR(1) model: y_t = a + b*y_{t-1}.
    Returns (a, b) or None if insufficient data.
    """
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 5:
        return None

    x_prev = y[:-1]
    y_curr = y[1:]

    # Least squares for y_curr = a + b*x_prev
    A = np.vstack([np.ones_like(x_prev), x_prev]).T
    (a, b), *_ = np.linalg.lstsq(A, y_curr, rcond=None)
    a = float(a)
    b = float(b)
    return a, b


def forecast_next(series: Sequence[float]) -> float:
    """One-step AR(1) forecast."""
    fit = _fit_ar1(series)
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if fit is None or y.size == 0:
        return float(y[-1]) if y.size else 0.0

    a, b = fit
    return float(a + b * float(y[-1]))


def time_to_threshold_ar1(series: Sequence[float], threshold: float = 1.5, max_steps: int = 200) -> Optional[float]:
    """
    Estimate time-to-threshold by iterating AR(1) forward until prediction >= threshold.
    Returns None if never crossed within max_steps or if AR(1) is not fit.
    """
    fit = _fit_ar1(series)
    y = np.asarray(list(series), dtype=float)
    y = y[np.isfinite(y)]
    if fit is None or y.size == 0:
        return None

    a, b = fit
    current = float(y[-1])
    if current >= threshold:
        return 0.0

    # Iterate until crossing or stability fails
    t = 0
    next_val = current
    while t < max_steps:
        next_val = float(a + b * next_val)
        t += 1
        if next_val >= threshold:
            return float(t)

        # If AR(1) is effectively decaying and won't cross, bail early
        if b <= 0:
            # Sequence will oscillate/decay without upward trend; safe to stop
            break

    return None


__all__ = ["forecast_next", "time_to_threshold_ar1"]

