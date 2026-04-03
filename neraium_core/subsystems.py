from __future__ import annotations

import numpy as np
from numpy.linalg import eigvals, eig


def spectral_clustering_subsystems(corr: np.ndarray, k: int = 3) -> list[list[int]]:
    """
    Lightweight subsystem discovery using eigenvector embedding.

    This is a practical approximation, not a full clustering framework.
    """
    corr = np.asarray(corr, dtype=float)

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] < 2:
        return []

    vals, vecs = eig(corr)
    idx = np.argsort(-np.abs(vals))[: min(k, corr.shape[0])]
    embedding = np.real(vecs[:, idx])

    labels = np.argmax(np.abs(embedding), axis=1)

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(i)

    return list(clusters.values())


def discover_subsystems(corr: np.ndarray, k: int = 3) -> list[list[int]]:
    """
    Backward-compatible subsystem discovery wrapper.

    The test suite expects:
    - for a single-sensor correlation matrix, no subsystems are discovered.
    """
    clusters = spectral_clustering_subsystems(corr, k=k)
    # Only subsystems with at least 2 sensors are meaningful for spectral measures.
    return [cluster for cluster in clusters if len(cluster) >= 2]


def subsystem_spectral_measures(corr: np.ndarray, k: int = 3) -> dict[str, object]:
    """
    Compute subsystem-local dominant spectral instability.

    Returns both the discovered clusters and the max subsystem instability.
    """
    corr = np.asarray(corr, dtype=float)

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1] or corr.shape[0] < 2:
        return {
            "clusters": [],
            "subsystem_instability": 0.0,
            "max_instability": 0.0,
            "subsystem_count": 0,
        }

    clusters = spectral_clustering_subsystems(corr, k=k)
    instabilities: list[float] = []

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        sub = corr[np.ix_(cluster, cluster)]
        vals = eigvals(sub)
        instabilities.append(float(np.max(np.abs(vals))))

    max_instability = float(max(instabilities)) if instabilities else 0.0

    return {
        "clusters": clusters,
        "subsystem_instability": max_instability,
        "max_instability": max_instability,
        "subsystem_count": len(clusters),
    }