from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class TrajectoryArchetype:
    centroid: np.ndarray
    count: int = 0
    escalation_events: int = 0


class TrajectoryArchetypeMemory:
    """Trajectory-level archetype memory using simple shape features."""

    def __init__(self, max_archetypes: int = 8) -> None:
        self.max_archetypes = max_archetypes
        self._store: dict[str, TrajectoryArchetype] = {}

    def _features(self, trajectory: list[list[float]]) -> np.ndarray:
        z = np.asarray(trajectory, dtype=float)
        if len(z) < 2:
            return np.zeros(8)
        diffs = np.diff(z, axis=0)
        norms = np.linalg.norm(z, axis=1)
        displacement = float(np.linalg.norm(z[-1] - z[0]))
        path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        mean_radius = float(np.mean(norms))
        end_radius = float(norms[-1])
        max_radius = float(np.max(norms))
        vel = np.linalg.norm(diffs, axis=1)
        vel_mean = float(np.mean(vel))
        vel_std = float(np.std(vel))
        return np.array([displacement, path_len, mean_radius, end_radius, max_radius, vel_mean, vel_std, float(len(z))])

    def _closest(self, f: np.ndarray) -> tuple[str | None, float]:
        if not self._store:
            return None, 0.0
        best, best_d = None, float("inf")
        for k, v in self._store.items():
            d = float(np.linalg.norm(f - v.centroid))
            if d < best_d:
                best, best_d = k, d
        sim = float(np.exp(-best_d))
        return best, sim

    def update(self, asset_id: str, trajectory: list[list[float]], transition, escalating: bool) -> dict:
        f = self._features(trajectory)
        name, sim = self._closest(f)
        if name is None or sim < 0.6:
            name = f"traj_{len(self._store)+1}"
            if len(self._store) >= self.max_archetypes:
                name = list(self._store.keys())[0]
                del self._store[name]
            self._store[name] = TrajectoryArchetype(f.copy())
            sim = 1.0
        entry = self._store[name]
        entry.count += 1
        lr = 1.0 / entry.count
        entry.centroid = (1 - lr) * entry.centroid + lr * f
        if escalating:
            entry.escalation_events += 1
        esc_freq = entry.escalation_events / max(1, entry.count)

        return {
            "current_trajectory_archetype": name,
            "similarity": round(sim, 4),
            "novelty_score": round(1.0 - sim, 4),
            "nearest_trajectory_archetypes": [
                {"name": k, "similarity": round(float(np.exp(-np.linalg.norm(f - v.centroid))), 4), "escalation_freq": round(v.escalation_events / max(1, v.count), 4)}
                for k, v in list(self._store.items())[:3]
            ],
            "forecast": {
                "likely_next_path_family": name,
                "trajectory_conditioned_escalation_probability": round(float(esc_freq), 4),
                "estimated_steps_to_critical_region": round(float(1.0 / (esc_freq + 1e-3)), 2),
                "uncertainty": round(float(1.0 / np.sqrt(max(2, entry.count))), 4),
            },
        }
