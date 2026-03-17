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


def _component_count(adjacency: np.ndarray) -> int:
    n = adjacency.shape[0]
    seen: set[int] = set()
    components = 0
    for root in range(n):
        if root in seen:
            continue
        components += 1
        stack = [root]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            neighbors = np.where(adjacency[node] > 0)[0].tolist()
            stack.extend(neighbors)
    return components


def graph_metrics(adjacency: ArrayLike) -> dict[str, float]:
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

    component_count = _component_count(adj) if n else 0

    return {
        "mean_degree": float(np.mean(degree) if n else 0.0),
        "density": density,
        "clustering": clustering,
        "component_count": float(component_count),
        "is_connected": 1.0 if component_count <= 1 else 0.0,
    }
