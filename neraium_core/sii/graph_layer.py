from __future__ import annotations

import numpy as np

from neraium_core.sii.types import GraphSnapshot, GraphState


def build_graph_snapshot(state: GraphState, *, threshold: float) -> GraphSnapshot:
    corr = np.asarray(state.adjacency, dtype=float)
    adj = (np.abs(corr) >= float(threshold)).astype(float)
    np.fill_diagonal(adj, 0.0)

    n = int(adj.shape[0]) if adj.ndim == 2 else 0
    edge_count = int(np.sum(adj))
    density = float(edge_count / max(1.0, float(n * (n - 1))))
    mean_abs_weight = float(np.mean(np.abs(corr[np.triu_indices(n, k=1)]))) if n > 1 else 0.0
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(corr)))) if n > 0 else 0.0

    deg = np.sum(adj, axis=1) if n > 0 else np.array([], dtype=float)
    degree_centrality = {
        state.feature_names[i]: float(deg[i] / max(1.0, float(n - 1)))
        for i in range(min(n, len(state.feature_names)))
    }

    lap = np.diag(np.sum(adj, axis=1)) - adj if n > 0 else np.zeros((0, 0), dtype=float)
    lap_trace = float(np.trace(lap)) if n > 0 else 0.0

    return GraphSnapshot(
        adjacency=adj,
        node_count=n,
        edge_count=edge_count,
        density=density,
        mean_abs_weight=mean_abs_weight,
        spectral_radius=spectral_radius,
        laplacian_trace=lap_trace,
        degree_centrality=degree_centrality,
    )


def graph_departure_score(current: GraphSnapshot, baseline: GraphSnapshot | None) -> float:
    if baseline is None or baseline.adjacency.shape != current.adjacency.shape:
        return 0.0

    l1_deformation = float(np.mean(np.abs(current.adjacency - baseline.adjacency)))
    density_shift = abs(float(current.density) - float(baseline.density))
    spectral_shift = abs(float(current.spectral_radius) - float(baseline.spectral_radius))
    return float(0.45 * l1_deformation + 0.30 * density_shift + 0.25 * spectral_shift)


def graph_state(
    current_corr: np.ndarray,
    baseline_adj: np.ndarray | None,
    threshold: float,
) -> GraphState:
    corr = np.asarray(current_corr, dtype=float)
    adj = (np.abs(corr) >= float(threshold)).astype(float)
    np.fill_diagonal(adj, 0.0)
    n = int(adj.shape[0]) if adj.ndim == 2 else 0
    edges = float(np.sum(adj))
    density = float(edges / max(1.0, float(n * (n - 1))))
    avg_degree = float(edges / max(1.0, float(n)))
    if baseline_adj is None or baseline_adj.shape != adj.shape:
        deformation = 0.0
    else:
        deformation = float(np.mean(np.abs(adj - baseline_adj)))
    _ = density
    _ = avg_degree
    _ = deformation
    return GraphState(
        adjacency=adj,
        feature_names=[f"s{i}" for i in range(n)],
    )
