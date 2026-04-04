from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


def _undirected_graph_is_connected(adj: np.ndarray) -> bool:
    """Return True iff the undirected graph has a single connected component.

    Uses a stack-based traversal over a boolean adjacency view to avoid matrix
    powers and repeated dense multiplications.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be square")
    n = int(adj.shape[0])
    if n <= 1:
        return True

    a = np.asarray(adj, dtype=float) > 0.0
    seen = np.zeros(n, dtype=bool)
    stack = [0]
    seen[0] = True
    seen_count = 1
    while stack:
        node = stack.pop()
        neighbors = np.flatnonzero(a[node] & ~seen)
        if neighbors.size == 0:
            continue
        seen[neighbors] = True
        seen_count += int(neighbors.size)
        stack.extend(neighbors.tolist())
        if seen_count == n:
            return True
    return False


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

    triangles = float(np.einsum("ij,jk,ki->", adj, adj, adj, optimize=True) / 6.0) if n >= 3 else 0.0
    triplets = float(np.sum(degree * (degree - 1)) / 2.0)
    clustering = float((3.0 * triangles / triplets) if triplets > 0 else 0.0)

    connected = float(_undirected_graph_is_connected(adj)) if n else 0.0

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
