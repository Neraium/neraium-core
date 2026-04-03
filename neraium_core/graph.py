from __future__ import annotations

from typing import Any

import numpy as np


ArrayLike = Any


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
        reachability = np.linalg.matrix_power(adj + np.eye(n), max(n - 1, 1))
        connected = float(np.all(reachability > 0))

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
