from __future__ import annotations

from typing import Any

import numpy as np

from neraium_core.graph import thresholded_adjacency
from neraium_core.spectral import spectral_radius


ArrayLike = Any


def _connected_components(adjacency: np.ndarray) -> list[list[int]]:
    n = adjacency.shape[0]
    seen = set()
    components: list[list[int]] = []

    for root in range(n):
        if root in seen:
            continue
        stack = [root]
        comp: list[int] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            neighbors = np.where(adjacency[node] > 0)[0].tolist()
            stack.extend(neighbors)
        components.append(sorted(comp))

    return components


def discover_subsystems(corr: ArrayLike, threshold: float = 0.7) -> list[list[int]]:
    adjacency = thresholded_adjacency(corr, threshold=threshold)
    return _connected_components(adjacency)


def subsystem_spectral_measures(corr: ArrayLike, threshold: float = 0.7) -> dict[str, float]:
    matrix = np.asarray(corr, dtype=float)
    components = discover_subsystems(matrix, threshold=threshold)
    if not components:
        return {"subsystem_count": 0.0, "max_subsystem_radius": 0.0, "subsystem_instability": 0.0}

    radii = []
    for component in components:
        block = matrix[np.ix_(component, component)]
        radii.append(spectral_radius(block))

    max_radius = float(max(radii) if radii else 0.0)
    return {
        "subsystem_count": float(len(components)),
        "max_subsystem_radius": max_radius,
        "subsystem_instability": max_radius,
    }
