from __future__ import annotations

from typing import Any

import numpy as np


def build_regime_signature(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Build a regime signature from per-signal mean and std."""
    return np.concatenate([np.asarray(mean, dtype=float), np.asarray(std, dtype=float)])


def regime_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between regime signatures."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def assign_regime(signature: np.ndarray, regimes: list[dict[str, Any]]) -> dict[str, float | str] | None:
    """Find the nearest known regime."""
    if not regimes:
        return None

    distances: list[tuple[float, str]] = []
    for regime in regimes:
        centroid = np.asarray(regime["signature"], dtype=float)
        distances.append((regime_distance(signature, centroid), str(regime["name"])))

    distances.sort(key=lambda x: x[0])
    nearest_distance, nearest_name = distances[0]
    return {"name": nearest_name, "distance": float(nearest_distance)}


def update_regime_library(
    signature: np.ndarray,
    regimes: list[dict[str, Any]],
    threshold: float = 2.0,
) -> list[dict[str, Any]]:
    """
    Update the regime library.

    If no regime exists, bootstrap one.
    If the nearest regime is farther than threshold, add a new regime.
    Otherwise keep the existing library unchanged.
    """
    if not regimes:
        regimes.append({"name": "regime_0", "signature": signature.tolist()})
        return regimes

    assigned = assign_regime(signature, regimes)
    if assigned is None or float(assigned["distance"]) > threshold:
        regimes.append(
            {
                "name": f"regime_{len(regimes)}",
                "signature": signature.tolist(),
            }
        )

    return regimes