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


def _trajectory_divergence_factor(projected: np.ndarray, radius: float) -> float:
    if projected.shape[0] < 4:
        return 1.0
    directions: dict[tuple[int, ...], set[tuple[int, int]]] = {}
    for i in range(projected.shape[0] - 1):
        src = _region_id(projected[i], bucket=radius)
        delta = projected[i + 1] - projected[i]
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-9:
            continue
        unit = delta / norm
        quantized = tuple(np.round(unit * 4.0).astype(int).tolist())
        directions.setdefault(src, set()).add(quantized)
    options = [len(v) for v in directions.values() if v]
    return float(np.mean(options)) if options else 1.0


def _local_successor_branching(projected: np.ndarray) -> float:
    if projected.shape[0] < 6:
        return 1.0
    deltas = projected[1:] - projected[:-1]
    step_scale = float(np.median(np.linalg.norm(deltas, axis=1))) + 1e-9
    radius = max(0.5, 1.5 * step_scale)
    branch_counts: list[int] = []
    for i in range(projected.shape[0] - 2):
        src = projected[i]
        dists = np.linalg.norm(projected[:-1] - src, axis=1)
        neighbors = np.where(dists <= radius)[0]
        if neighbors.size < 2:
            continue
        successors = {_region_id(projected[n + 1], bucket=0.8) for n in neighbors}
        branch_counts.append(len(successors))
    return float(np.mean(branch_counts)) if branch_counts else 1.0


def compute_state_graph(
    path: list[np.ndarray],
    window: int = 16,
    geometry: dict[str, float] | None = None,
    state_space_statistics: dict[str, float] | None = None,
) -> dict[str, float | dict[str, int]]:
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

    history = np.vstack([np.asarray(v, dtype=float) for v in path])
    projected = _pca_reduce(history, dims=2)
    nodes = [_region_id(p) for p in projected]
    edges = list(zip(nodes[:-1], nodes[1:]))

    node_set = set(nodes)
    edge_counts: Counter[tuple[tuple[int, ...], tuple[int, ...]]] = Counter(edges)
    successors: dict[tuple[int, ...], set[tuple[int, ...]]] = {}
    for a, b in edges:
        if a != b:
            successors.setdefault(a, set()).add(b)

    node_count = len(node_set)
    edge_count = len(edge_counts)
    successor_counts = [len(v) for v in successors.values() if v]
    structural_branching = float(np.mean(successor_counts)) if successor_counts else 0.0
    recent = projected[-max(4, window) :]
    divergence_branching = _trajectory_divergence_factor(recent, radius=0.7)
    local_branching = _local_successor_branching(recent)
    probs = np.asarray(list(edge_counts.values()), dtype=float)
    probs = probs / (np.sum(probs) + 1e-12)
    entropy = float(-np.sum([p * math.log(p + 1e-12) for p in probs])) if probs.size else 0.0
    entropy_norm = float(entropy / max(1e-9, math.log(max(2, probs.size)))) if probs.size > 1 else 0.0

    divergence = float(max(0.0, 1.0 - (float(np.max(probs)) if probs.size else 0.0)))
    directional_consistency = float(np.clip((geometry or {}).get("directional_consistency", 1.0), 0.0, 1.0))
    directional_inconsistency = float(1.0 - directional_consistency)
    anisotropy = float(np.clip((state_space_statistics or {}).get("anisotropy", 1.0), 0.0, 1.0))
    multi_directionality = float(1.0 - anisotropy)

    base_branching = 0.35 * structural_branching + 0.25 * divergence_branching + 0.20 * local_branching
    uncertainty_gain = (
        0.45 * entropy_norm
        + 0.35 * divergence
        + 0.25 * directional_inconsistency
        + 0.20 * multi_directionality
    )
    temporal_variation = 0.15 * float(np.std(np.linalg.norm(np.diff(recent, axis=0), axis=1))) if recent.shape[0] > 2 else 0.0
    branching_factor = max(0.0, base_branching + uncertainty_gain + temporal_variation)

    revisit_rate = float(1.0 - (len(set(nodes)) / max(1, len(nodes))))
    dominant_path_prob = float(np.max(probs)) if probs.size else 0.0
    path_commitment = dominant_path_prob
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
