"""Compatibility shim for forecast helpers.

Canonical implementation lives in ``neraium_core.forecast_models``.
"""

from __future__ import annotations

from collections.abc import Sequence

from neraium_core.forecast_models import _fit_ar1, forecast_next, time_to_threshold_ar1


def fit_ar1(series: Sequence[float]) -> tuple[float, float]:
    """Back-compat AR(1) fit returning ``(phi, intercept)``."""
    fit = _fit_ar1(series)
    if fit is None:
        values = list(series)
        return 0.0, float(values[-1]) if values else 0.0
    intercept, phi = fit
    return float(phi), float(intercept)


__all__ = ["fit_ar1", "forecast_next", "time_to_threshold_ar1"]
