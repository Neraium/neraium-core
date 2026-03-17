from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def _as_2d_array(values: ArrayLike) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array-like structure")
    if array.shape[0] < 2:
        raise ValueError("At least two observations are required")
    return array


def correlation_matrix(observations: ArrayLike) -> np.ndarray:
    """Compute a correlation matrix from row-wise observations."""
    data = _as_2d_array(observations)
    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def relational_structure(corr: ArrayLike) -> dict[str, np.ndarray | float]:
    """Extract compact relational observables from a correlation matrix."""
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")

    magnitude = np.abs(matrix)
    np.fill_diagonal(magnitude, 0.0)
    centrality = magnitude.mean(axis=1)
    relational_energy = float(magnitude.sum() / max(matrix.shape[0] * (matrix.shape[0] - 1), 1))

    return {
        "centrality": centrality,
        "relational_energy": relational_energy,
    }


def relational_drift(current_corr: ArrayLike, baseline_corr: ArrayLike) -> dict[str, float]:
    """Measure baseline-relative drift between two correlation structures."""
    current = np.asarray(current_corr, dtype=float)
    baseline = np.asarray(baseline_corr, dtype=float)
    if current.shape != baseline.shape:
        raise ValueError("Current and baseline correlation matrices must share shape")

    delta = current - baseline
    fro_norm = float(np.linalg.norm(delta, ord="fro"))
    mean_abs = float(np.mean(np.abs(delta)))
    max_abs = float(np.max(np.abs(delta)))

    baseline_scale = float(np.linalg.norm(baseline, ord="fro"))
    relative_drift = fro_norm / baseline_scale if baseline_scale > 0 else fro_norm

    return {
        "frobenius_drift": fro_norm,
        "mean_abs_drift": mean_abs,
        "max_abs_drift": max_abs,
        "relative_drift": relative_drift,
    }
