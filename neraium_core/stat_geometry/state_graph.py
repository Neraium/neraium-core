from __future__ import annotations

import math
from collections import Counter
import numpy as np


def _pca_reduce(states: np.ndarray, dims: int = 2) -> np.ndarray:
    if states.shape[0] == 0:
        return np.zeros((0, dims), dtype=float)
    centered = states - np.mean(states, axis=0, keepdims=True)
    cov = np.cov(centered.T) if states.shape[0] > 1 else np.eye(states.shape[1], dtype=float)
    cov = np.atleast_2d(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order[: min(dims, eigvecs.shape[1])]]
    out = centered @ basis
    if out.shape[1] < dims:
        out = np.pad(out, ((0, 0), (0, dims - out.shape[1])), mode="constant")
    return out


def _region_id(point: np.ndarray, bucket: float = 0.8) -> tuple[int, ...]:
    q = np.floor(np.asarray(point, dtype=float) / max(bucket, 1e-9)).astype(int)
    return tuple(int(v) for v in q.tolist())


def compute_state_graph(path: list[np.ndarray], window: int = 16) -> dict[str, float | dict[str, int]]:
    if len(path) < 2:
        return {
            "node_count": 1,
            "edge_count": 0,
            "branching_factor": 0.0,
            "transition_entropy": 0.0,
            "revisit_rate": 0.0,
            "path_commitment_score": 0.0,
            "graph_divergence_score": 0.0,
            "graph_density": 0.0,
            "region_histogram": {},
        }

    tail = np.vstack([np.asarray(v, dtype=float) for v in path[-max(3, window):]])
    projected = _pca_reduce(tail, dims=2)
    nodes = [_region_id(p) for p in projected]
    edges = list(zip(nodes[:-1], nodes[1:]))

    node_set = set(nodes)
    edge_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter(edges)
    out_degree: Counter[tuple[int, ...]] = Counter()
    for a, b in edges:
        if a != b:
            out_degree[a] += 1

    node_count = len(node_set)
    edge_count = len(edge_counts)
    branching_factor = float(np.mean(list(out_degree.values()))) if out_degree else 0.0

    probs = np.asarray(list(edge_counts.values()), dtype=float)
    probs = probs / (np.sum(probs) + 1e-12)
    entropy = float(-np.sum([p * math.log(p + 1e-12) for p in probs])) if probs.size else 0.0
    entropy_norm = float(entropy / max(1e-9, math.log(max(2, probs.size)))) if probs.size > 1 else 0.0

    revisit_rate = float(1.0 - (len(set(nodes)) / max(1, len(nodes))))
    dominant_path_prob = float(np.max(probs)) if probs.size else 0.0
    path_commitment = dominant_path_prob
    divergence = float(max(0.0, 1.0 - dominant_path_prob))
    density = float(edge_count / max(1, node_count * max(1, node_count - 1)))

    hist = Counter(nodes)
    return {
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "branching_factor": round(branching_factor, 6),
        "transition_entropy": round(entropy_norm, 6),
        "revisit_rate": round(revisit_rate, 6),
        "path_commitment_score": round(path_commitment, 6),
        "graph_divergence_score": round(divergence, 6),
        "graph_density": round(density, 6),
        "region_histogram": {str(k): int(v) for k, v in hist.items()},
    }
