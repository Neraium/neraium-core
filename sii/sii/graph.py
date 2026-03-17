from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def build_graph(corr: pd.DataFrame, tau: float = 0.6) -> tuple[np.ndarray, nx.Graph]:
    abs_corr = np.abs(corr.values)
    adj = (abs_corr > tau).astype(int)
    np.fill_diagonal(adj, 0)
    graph = nx.from_numpy_array(adj)
    graph = nx.relabel_nodes(graph, dict(enumerate(corr.columns)))
    return adj, graph


def graph_metrics(graph: nx.Graph) -> dict[str, float]:
    n = graph.number_of_nodes()
    if n == 0:
        return {
            "mean_connectivity": 0.0,
            "density": 0.0,
            "connected_components": 0,
            "clustering_coefficient": 0.0,
        }
    degrees = [d for _, d in graph.degree()]
    return {
        "mean_connectivity": float(np.mean(degrees) if degrees else 0.0),
        "density": float(nx.density(graph)),
        "connected_components": int(nx.number_connected_components(graph)),
        "clustering_coefficient": float(nx.average_clustering(graph) if n > 1 else 0.0),
    }
