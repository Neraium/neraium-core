from __future__ import annotations
import numpy as np


def build_regime_signature(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.concatenate([mean, std])


def regime_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def assign_regime(signature: np.ndarray, regimes: list[dict]) -> tuple[str, float] | None:
    if not regimes:
        return None

    distances = []
    for r in regimes:
        centroid = np.asarray(r["signature"], dtype=float)
        d = regime_distance(signature, centroid)
        distances.append((d, r["name"]))

    distances.sort(key=lambda x: x[0])
    return distances[0][1], distances[0][0]


def update_regime_library(signature: np.ndarray, regimes: list[dict], threshold: float = 2.0):
    if not regimes:
        regimes.append({"name": "regime_0", "signature": signature.tolist()})
        return regimes

    assigned = assign_regime(signature, regimes)

    if assigned is None or assigned[1] > threshold:
        regimes.append({
            "name": f"regime_{len(regimes)}",
            "signature": signature.tolist()
        })

    return regimes