from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def eigendecomposition(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    if values.size == 0:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    eigenvalues, eigenvectors = np.linalg.eigh(safe_values)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def spectral_radius(matrix: ArrayLike) -> float:
    eigenvalues, _ = eigendecomposition(matrix)
    return float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0


def spectral_gap(matrix: ArrayLike) -> float:
    eigenvalues, _ = eigendecomposition(matrix)
    if eigenvalues.size < 2:
        return 0.0
    return float(eigenvalues[0] - eigenvalues[1])


def dominant_mode_loading(matrix: ArrayLike) -> dict[str, list[float] | float]:
    eigenvalues, eigenvectors = eigendecomposition(matrix)
    if eigenvalues.size == 0:
        return {"dominant_eigenvalue": 0.0, "dominant_eigenvector": []}
    return {
        "dominant_eigenvalue": float(eigenvalues[0]),
        "dominant_eigenvector": [float(v) for v in eigenvectors[:, 0]],
    }
