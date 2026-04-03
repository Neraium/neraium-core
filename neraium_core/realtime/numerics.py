"""Numerical stability helpers for incremental structural paths."""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def safe_div(num: np.ndarray | float, den: np.ndarray | float, *, fill: float = 0.0) -> np.ndarray:
    """Elementwise division with epsilon floor on |den|."""
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    b = np.where(np.abs(b) < EPS, np.sign(b) * EPS + (EPS if b >= 0 else -EPS), b)
    out = a / b
    out = np.where(np.isfinite(out), out, fill)
    return out


def clip_symmetric(m: np.ndarray, limit: float) -> np.ndarray:
    x = np.asarray(m, dtype=np.float64)
    return np.clip(x, -limit, limit)


def sanitize_finite(x: np.ndarray, *, fill: float = 0.0) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return np.nan_to_num(a, nan=fill, posinf=fill, neginf=fill)


def regularize_covariance(cov: np.ndarray, *, ridge: float | None = None) -> np.ndarray:
    """Symmetric PSD-ish regularization for covariance matrices."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        return c
    n = c.shape[0]
    r = float(ridge if ridge is not None else max(EPS, 1e-9 * float(np.trace(c)) / max(n, 1)))
    c = 0.5 * (c + c.T)
    c = c + r * np.eye(n, dtype=np.float64)
    return c
