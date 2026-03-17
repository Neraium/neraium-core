from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def eigendecomposition(matrix: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def spectral_radius(matrix: ArrayLike) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    if values.shape[0] < 2:
        return 0.0

    eigenvalues, _ = eigendecomposition(values)
    if eigenvalues.size == 0:
        return 0.0
    return float(np.max(np.abs(eigenvalues)))


def spectral_gap(matrix: ArrayLike) -> float:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Matrix must be square")
    if values.shape[0] < 2:
        return 0.0

    eigenvalues, _ = eigendecomposition(values)
    if eigenvalues.size < 2:
        return 0.0
    return float(eigenvalues[0] - eigenvalues[1])
