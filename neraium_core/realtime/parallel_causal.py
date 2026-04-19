"""Parallel computation of causal analysis metrics.

Provides multi-core implementations of expensive causal operations
like Granger causality matrix computation.
"""
from __future__ import annotations

from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Optional

import numpy as np


def _compute_granger_pair(i: int, j: int, X: np.ndarray, lag: int = 1) -> tuple[int, int, float]:
    """Compute Granger-style causality score for a single pair (i, j).

    Returns (i, j, score) tuple for sparse filling.
    """
    n = X.shape[1]
    if i == j or i >= n or j >= n:
        return i, j, 0.0

    x = X[:-lag, i]
    y = X[lag:, j]

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 5:
        return i, j, 0.0

    denom = float(np.dot(x, x))
    if abs(denom) < 1e-12:
        return i, j, 0.0

    beta = float(np.dot(x, y) / denom)
    pred = beta * x
    error = float(np.mean((y - pred) ** 2))

    return i, j, 1.0 / (error + 1e-6)


def granger_causality_matrix_parallel(
    X: np.ndarray,
    lag: int = 1,
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """Compute Granger-style causality matrix using parallel workers.

    Args:
        X: Time-series data matrix of shape (T, N)
        lag: Lag for causality computation
        n_workers: Number of worker processes. If None, uses CPU count.

    Returns:
        Causality matrix of shape (N, N)
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[0] <= lag or X.shape[1] < 2:
        return np.zeros((X.shape[1] if X.ndim == 2 else 0, X.shape[1] if X.ndim == 2 else 0))

    n = X.shape[1]
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    # Use single-threaded version for small matrices
    if n < 4 or n_workers < 2:
        return _granger_causality_matrix_serial(X, lag)

    # Generate all (i, j) pairs
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    # Parallel computation
    compute_fn = partial(_compute_granger_pair, X=X, lag=lag)
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(compute_fn, [(p[0], p[1]) for p in pairs])

    # Fill matrix
    C = np.zeros((n, n), dtype=float)
    for i, j, score in results:
        C[i, j] = score

    return C


def _granger_causality_matrix_serial(X: np.ndarray, lag: int = 1) -> np.ndarray:
    """Serial version of Granger causality (for small matrices or fallback)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] <= lag or X.shape[1] < 2:
        return np.zeros((X.shape[1] if X.ndim == 2 else 0, X.shape[1] if X.ndim == 2 else 0))

    n = X.shape[1]
    C = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            x = X[:-lag, i]
            y = X[lag:, j]

            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            if len(x) < 5:
                continue

            denom = float(np.dot(x, x))
            if abs(denom) < 1e-12:
                continue

            beta = float(np.dot(x, y) / denom)
            pred = beta * x
            error = float(np.mean((y - pred) ** 2))

            C[i, j] = 1.0 / (error + 1e-6)

    return C
