from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def _undirected_graph_is_connected(adj: np.ndarray) -> bool:
    """Return True iff the undirected graph has a single connected component.

    Replaces ``(A+I)^(n-1)`` (O(n^3 log n) matrix powers) with repeated
    ``r @ A`` frontier expansion (BLAS, O(diameter·n^2), typically << n steps).
    Matches the legacy Boolean ``np.all(R > 0)`` reachability on symmetric 0/1
    adjacency from ``thresholded_adjacency`` (verified on random graphs).
    """
    n = int(adj.shape[0])
    if n <= 1:
        return True
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be square")
    A = (np.asarray(adj, dtype=float) > 0.0).astype(np.float64)
    r = np.zeros(n, dtype=np.float64)
    r[0] = 1.0
    for _ in range(n):
        spread = (r @ A) > 0.0
        nxt = np.maximum(r, spread.astype(np.float64))
        if np.array_equal(nxt, r):
            break
        r = nxt
    return bool(np.all(r > 0.0))


def thresholded_adjacency(corr: ArrayLike, threshold: float = 0.6) -> np.ndarray:
    matrix = np.asarray(corr, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square")
    adj = (np.abs(matrix) >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj


def graph_metrics(adjacency: ArrayLike, corr: ArrayLike | None = None) -> dict[str, float]:
    adj = np.asarray(adjacency, dtype=float)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    n = adj.shape[0]
    degree = adj.sum(axis=1)
    max_edges = n * (n - 1)
    density = float(adj.sum() / max_edges) if max_edges else 0.0

    triangles = float(np.trace(np.linalg.matrix_power(adj, 3)) / 6.0) if n >= 3 else 0.0
    triplets = float(np.sum(degree * (degree - 1)) / 2.0)
    clustering = float((3.0 * triangles / triplets) if triplets > 0 else 0.0)

    connected = 0.0
    if n:
        # Small graphs: BLAS matrix_power is extremely cheap; large n: avoid (A+I)^(n-1).
        if n <= 24:
            reachability = np.linalg.matrix_power(adj + np.eye(n), max(n - 1, 1))
            connected = float(np.all(reachability > 0))
        else:
            connected = float(_undirected_graph_is_connected(adj))

    metrics = {
        "mean_degree": float(np.mean(degree) if n else 0.0),
        "density": density,
        "clustering": clustering,
        "connectivity": connected,
    }

    if corr is not None:
        corr_matrix = np.asarray(corr, dtype=float)
        if corr_matrix.shape != adj.shape:
            raise ValueError("corr must have the same shape as adjacency")
        metrics["mean_absolute_connectivity"] = float(np.mean(np.abs(corr_matrix - np.eye(n))) if n else 0.0)

    return metrics
