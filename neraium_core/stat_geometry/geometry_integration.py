from __future__ import annotations

from collections import defaultdict
import numpy as np

from .state_graph import compute_state_graph
from .state_space import StateSpaceProjector
from .state_statistics import compute_state_statistics
from .trajectory_geometry import compute_trajectory_geometry


class StatisticalGeometryLayer:
    """Entity-aware deterministic geometry/statistical graph layer."""

    def __init__(self, max_history: int = 256, graph_window: int = 16, stats_window: int = 12) -> None:
        self.max_history = max_history
        self.graph_window = graph_window
        self.stats_window = stats_window
        self.entity_paths: dict[str, list[np.ndarray]] = defaultdict(list)

    def _fleet_summary(self) -> dict[str, float]:
        entities = sorted(self.entity_paths.keys())
        if len(entities) < 2:
            return {
                "fleet_entity_count": len(entities),
                "fleet_mean_pairwise_distance": 0.0,
                "fleet_shared_region_ratio": 1.0,
            }
        latest = [self.entity_paths[e][-1] for e in entities if self.entity_paths[e]]
        if len(latest) < 2:
            return {
                "fleet_entity_count": len(entities),
                "fleet_mean_pairwise_distance": 0.0,
                "fleet_shared_region_ratio": 1.0,
            }
        dists: list[float] = []
        for i in range(len(latest)):
            for j in range(i + 1, len(latest)):
                dists.append(float(np.linalg.norm(latest[i] - latest[j])))
        lens = [len(self.entity_paths[e]) for e in entities]
        shared_ratio = float(min(lens) / max(1, max(lens)))
        return {
            "fleet_entity_count": len(entities),
            "fleet_mean_pairwise_distance": round(float(np.mean(dists)) if dists else 0.0, 6),
            "fleet_shared_region_ratio": round(shared_ratio, 6),
        }

    @staticmethod
    def _explanations(geometry: dict[str, float], stats: dict[str, float], graph: dict[str, float | dict[str, int]]) -> dict[str, str]:
        branch_evidence = (
            geometry.get("curvature", 0.0) > 0.2 and float(graph.get("graph_divergence_score", 0.0)) > 0.35
        )
        lock_in_evidence = (
            stats.get("state_contraction_score", 0.0) > 0.25 and float(graph.get("path_commitment_score", 0.0)) > 0.55
        )
        drift_vs_noise = (
            "persistent directional drift"
            if geometry.get("directional_consistency", 0.0) >= 0.65
            else "diffuse wandering/noise-like movement"
        )
        return {
            "trajectory": (
                f"Trajectory motion shows directional consistency={geometry.get('directional_consistency', 0.0):.2f} "
                f"and curvature={geometry.get('curvature', 0.0):.2f}."
            ),
            "branching": "Geometric divergence suggests branching" if branch_evidence else "Branching evidence is limited",
            "lock_in": "Reachable state contraction and path commitment suggest lock-in" if lock_in_evidence else "No strong lock-in geometry yet",
            "drift_vs_noise": drift_vs_noise,
            "horizon": (
                f"Expansion={stats.get('state_expansion_score', 0.0):.2f}, contraction={stats.get('state_contraction_score', 0.0):.2f} "
                "indicates how quickly the state approaches unstable regions."
            ),
        }

    def update(self, *, entity_id: str, matrix: np.ndarray, representation_mode: str) -> dict[str, object]:
        mode_map = {
            "raw": "raw",
            "raw_only": "raw",
            "residualized": "residualized",
            "dynamic": "dynamic",
            "dynamic_only": "dynamic",
            "combined": "combined",
        }
        projector = StateSpaceProjector(mode=mode_map.get(str(representation_mode).lower(), "combined"))
        frame = projector.project_window(matrix)

        path = self.entity_paths[str(entity_id)]
        path.append(frame.state_vector)
        if len(path) > self.max_history:
            del path[:-self.max_history]

        geometry = compute_trajectory_geometry(path, window=self.stats_window)
        stats = compute_state_statistics(path, window=self.stats_window)
        graph = compute_state_graph(path, window=self.graph_window)
        return {
            "geometry": geometry,
            "state_space_statistics": stats,
            "state_graph": graph,
            "geometry_explanations": self._explanations(geometry, stats, graph),
            "fleet_geometry": self._fleet_summary(),
            "state_space": {
                "representation_mode": frame.mode,
                "state_vector_dim": int(frame.state_vector.size),
                "feature_group_dims": {k: int(v.size) for k, v in frame.feature_groups.items()},
            },
        }
