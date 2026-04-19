from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArchetypeCentroid:
    centroid: np.ndarray
    count: int = 0
    escalation_events: int = 0


class StructuralArchetypeMemory:
    """Online archetype memory for cross-system latent trajectories."""

    def __init__(self, max_archetypes: int = 8) -> None:
        self.max_archetypes = int(max_archetypes)
        self._store: dict[str, ArchetypeCentroid] = {}
        self._asset_archetype: dict[str, str] = {}

    def _closest(self, z: np.ndarray) -> tuple[str | None, float]:
        if not self._store:
            return None, 0.0
        best_name = None
        best_d = float("inf")
        for name, entry in self._store.items():
            d = float(np.linalg.norm(z - entry.centroid))
            if d < best_d:
                best_name, best_d = name, d
        similarity = float(np.exp(-best_d))
        return best_name, similarity

    def update(self, asset_id: str, embedding: list[float], escalating: bool) -> dict:
        z = np.asarray(embedding, dtype=float)
        name, similarity = self._closest(z)
        if name is None or similarity < 0.55:
            name = f"archetype_{len(self._store) + 1}"
            if len(self._store) >= self.max_archetypes:
                # replace sparsest cluster
                replace = sorted(self._store.items(), key=lambda kv: kv[1].count)[0][0]
                del self._store[replace]
                name = replace
            self._store[name] = ArchetypeCentroid(centroid=z.copy(), count=0, escalation_events=0)
            similarity = 1.0

        entry = self._store[name]
        entry.count += 1
        lr = 1.0 / float(entry.count)
        entry.centroid = (1.0 - lr) * entry.centroid + lr * z
        if escalating:
            entry.escalation_events += 1
        self._asset_archetype[str(asset_id)] = name

        escalation_frequency = float(entry.escalation_events / max(1, entry.count))
        sorted_matches = sorted(
            [
                {
                    "archetype": k,
                    "similarity": round(float(np.exp(-np.linalg.norm(z - v.centroid))), 4),
                    "historical_escalation_frequency": round(float(v.escalation_events / max(1, v.count)), 4),
                    "support": int(v.count),
                }
                for k, v in self._store.items()
            ],
            key=lambda item: item["similarity"],
            reverse=True,
        )[:3]

        return {
            "current_archetype": name,
            "similarity": round(float(similarity), 4),
            "historical_escalation_frequency": round(escalation_frequency, 4),
            "nearest_archetypes": sorted_matches,
        }
