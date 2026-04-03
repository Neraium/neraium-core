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


def normalize_window(observations: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-window z-score normalization with missing and zero-variance guards."""
    data = _as_2d_array(observations)
    means = np.nanmean(data, axis=0)
    means = np.nan_to_num(means, nan=0.0)

    centered = data - means
    std = np.nanstd(data, axis=0)
    std = np.nan_to_num(std, nan=0.0)
    safe_std = np.where(std <= 1e-12, 1.0, std)

    z = centered / safe_std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z, means, std


def correlation_matrix(observations: ArrayLike) -> np.ndarray:
    """Compute correlation geometry R_t from row-wise observations."""
    data = _as_2d_array(observations)
    corr = np.corrcoef(data, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def structural_drift(current_corr: ArrayLike, baseline_corr: ArrayLike, norm: str = "fro") -> float:
    """Baseline-relative structural drift D_t = ||R_t - R_0||."""
    current = np.asarray(current_corr, dtype=float)
    baseline = np.asarray(baseline_corr, dtype=float)
    if current.shape != baseline.shape or current.ndim != 2 or current.shape[0] != current.shape[1]:
        raise ValueError("current and baseline correlation matrices must be equally-shaped square matrices")

    delta = np.nan_to_num(current - baseline, nan=0.0, posinf=0.0, neginf=0.0)
    if norm == "mae":
        return float(np.mean(np.abs(delta)))
    if norm == "fro":
        return float(np.linalg.norm(delta, ord="fro"))
    raise ValueError("norm must be 'fro' or 'mae'")


def signal_structural_importance(corr: ArrayLike) -> np.ndarray:
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    n = matrix.shape[0]
    return np.mean(np.abs(matrix), axis=1) if n else np.array([], dtype=float)


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
