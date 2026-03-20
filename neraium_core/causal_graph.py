from __future__ import annotations

import numpy as np

# Compatibility module: the codebase contains `casual_graph` (typo) but other
# modules import `causal_graph`. Provide the expected API.


def causal_adjacency(C: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    A = np.array(C, dtype=float, copy=True)
    A[np.abs(A) < threshold] = 0.0
    np.fill_diagonal(A, 0.0)
    return A


def causal_graph_metrics(C: np.ndarray, threshold: float = 0.1) -> dict[str, float | list[int]]:
    A = causal_adjacency(C, threshold=threshold)

    if A.size == 0:
        return {
            "density": 0.0,
            "asymmetry": 0.0,
            "dominant_sources": [],
        }

    density = float(np.mean(np.abs(A) > 0))
    asymmetry = float(np.mean(np.abs(A - A.T)))
    outbound = np.sum(np.abs(A), axis=1)
    dominant_sources = np.argsort(-outbound)[:3].tolist()

    return {
        "density": density,
        "asymmetry": asymmetry,
        "dominant_sources": dominant_sources,
    }


def causal_root_cause_chains(
    C: np.ndarray,
    sensor_names: list[str],
    *,
    threshold: float = 0.1,
    max_depth: int = 3,
    chain_count: int = 2,
) -> list[dict[str, object]]:
    """
    Build propagation-like root-cause chains from an observational causal matrix.

    This is a *proxy narrative* over the Granger-style causal weights:
    - Directed edge i->j exists when |C[i,j]| >= threshold
    - Starting nodes are chosen by outbound strength
    - Each next hop is the strongest outgoing neighbor from the current node

    Returns a list of chains with node names and edge weights for explanation.
    """
    if C is None:
        return []

    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.size == 0:
        return []

    n = C.shape[0]
    if n < 2:
        return []

    if len(sensor_names) < n:
        # Fall back to generic node labels.
        names = [sensor_names[i] if i < len(sensor_names) else f"node_{i}" for i in range(n)]
    else:
        names = sensor_names[:n]

    max_depth = max(1, int(max_depth))
    chain_count = max(1, int(chain_count))

    absC = np.abs(C)
    # Outbound strength for starting nodes.
    outbound = np.sum(absC, axis=1)
    start_order = np.argsort(-outbound)[: min(chain_count, n)]

    # Build thresholded boolean reachability / allowed edges.
    allowed = absC >= float(threshold)
    np.fill_diagonal(allowed, False)

    chains: list[dict[str, object]] = []
    for s in start_order:
        chain_nodes: list[str] = [names[int(s)]]
        chain_edges: list[dict[str, object]] = []

        current = int(s)
        visited = {current}
        for _ in range(max_depth - 1):
            # Candidates: allowed outgoing edges not already in chain.
            cand_mask = allowed[current, :].copy()
            if visited:
                for v in visited:
                    cand_mask[v] = False
            if not bool(np.any(cand_mask)):
                break

            cand_idx = np.where(cand_mask)[0]
            # Choose strongest outgoing edge.
            next_i = int(cand_idx[int(np.argmax(absC[current, cand_idx]))])
            w = float(absC[current, next_i])

            chain_edges.append(
                {
                    "src": names[current],
                    "dst": names[next_i],
                    "edge_weight_abs": w,
                }
            )
            chain_nodes.append(names[next_i])
            visited.add(next_i)
            current = next_i

        chain_score = float(np.mean([float(e["edge_weight_abs"]) for e in chain_edges])) if chain_edges else 0.0
        chains.append(
            {
                "chain_nodes": chain_nodes,
                "chain_edges": chain_edges,
                "chain_score": chain_score,
            }
        )

    # Sort by chain_score desc and return top chain_count
    chains.sort(key=lambda x: float(x.get("chain_score", 0.0)), reverse=True)
    return chains[:chain_count]


def causal_propagation_spread(
    C: np.ndarray,
    *,
    threshold: float = 0.1,
    max_steps: int = 2,
    top_k: int = 3,
) -> dict[str, object]:
    """
    Propagation-aware causal proxy.

    Uses the Granger-style causal matrix C as a directed weighted adjacency
    (edge i->j exists when |C[i,j]| >= threshold). It then estimates which
    sources can reach other nodes within `max_steps` hops.

    This is still an observational proxy (not intervention-level causality),
    but it upgrades "correlation-only" reasoning into "propagation" reasoning.
    """
    if C is None:
        return {"top_sources": [], "spread_scores": [], "top_pairs": []}

    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.size == 0:
        return {"top_sources": [], "spread_scores": [], "top_pairs": []}

    n = C.shape[0]
    if n < 2:
        return {"top_sources": [], "spread_scores": [], "top_pairs": []}

    max_steps = max(1, int(max_steps))
    top_k = max(1, int(top_k))

    # Directed boolean adjacency: preserve direction (i->j) while thresholding by magnitude.
    adj_bool = np.abs(C) >= float(threshold)
    adj_bool = adj_bool & (~np.eye(n, dtype=bool))
    adj_int = adj_bool.astype(int)

    reach = adj_bool.copy()
    mat = adj_int.copy()
    for _ in range(max_steps - 1):
        mat = mat @ adj_int
        reach |= mat > 0

    # Never count self-reach.
    np.fill_diagonal(reach, False)

    spread = reach.sum(axis=1).astype(float)  # number of reachable nodes within max_steps
    denom = max(1, n - 1)
    spread_norm = spread / float(denom)

    order = np.argsort(-spread_norm)
    top_sources_idx = [int(i) for i in order[:top_k]]

    # For each top source, pick the reachable target with the strongest direct edge magnitude.
    absC = np.abs(C)
    top_pairs: list[dict[str, int | float]] = []
    for src in top_sources_idx:
        reachable = reach[src, :]
        if not bool(np.any(reachable)):
            continue
        # Mask unreachable and self, then pick argmax by edge magnitude.
        mask = reachable.astype(bool)
        masked_abs = absC[src, :].copy()
        masked_abs[~mask] = -np.inf
        dst = int(np.nanargmax(masked_abs))
        weight = float(absC[src, dst]) if np.isfinite(masked_abs[dst]) else 0.0
        top_pairs.append({"src_idx": src, "dst_idx": dst, "edge_weight": weight})

    return {
        "top_sources": top_sources_idx,
        "spread_scores": [float(x) for x in spread_norm.tolist()],
        "top_pairs": top_pairs,
    }


__all__ = ["causal_graph_metrics", "causal_propagation_spread", "causal_root_cause_chains"]

