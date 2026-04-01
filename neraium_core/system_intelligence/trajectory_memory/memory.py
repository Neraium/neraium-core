from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from neraium_core.system_intelligence.trajectory_archetypes.archetypes import TrajectorySegmentEncoder
from neraium_core.system_intelligence.trajectory_clustering import TrajectoryFamilyClusterer


class CrossSystemTrajectoryMemory:
    """Cross-asset memory of aligned trajectory segments and trajectory-family statistics."""

    def __init__(
        self,
        *,
        window_size: int = 10,
        max_archetypes: int = 14,
        min_similarity_for_match: float = 0.52,
    ) -> None:
        self.encoder = TrajectorySegmentEncoder(window_size=window_size)
        self.max_archetypes = int(max_archetypes)
        self.min_similarity_for_match = float(min_similarity_for_match)
        self._clusterer = TrajectoryFamilyClusterer(
            max_families=max_archetypes,
            min_similarity_for_match=min_similarity_for_match,
        )
        self._asset_history: dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=240))

    def update(
        self,
        *,
        asset_id: str,
        embedding: list[float],
        escalating: bool,
        in_critical_region: bool,
        phase_shift: bool,
        mechanism_names: list[str],
    ) -> dict[str, object]:
        z = np.asarray(embedding, dtype=float)
        hist = self._asset_history[str(asset_id)]
        hist.append(z)
        trajectory = [x.astype(float).tolist() for x in hist]
        segment = self.encoder.build_segment(trajectory)
        if segment is None:
            return {
                "status": "warming",
                "current_trajectory_archetype": None,
                "current_trajectory_family": None,
                "nearest_trajectory_archetypes": [],
                "nearest_trajectory_families": [],
                "family_similarity": 0.5,
                "trajectory_similarity": 0.5,
                "novelty_score": 0.5,
                "novelty_classification": "insufficient_history",
            }

        family, nearest, similarity = self._clusterer.assign(
            segment=segment.points,
            inferred_label=segment.path_family,
            escalating=bool(escalating),
            in_critical_region=bool(in_critical_region),
            phase_shift=bool(phase_shift),
            mechanism_names=mechanism_names,
        )
        novelty_score = max(0.0, min(1.0, 1.0 - similarity))
        novelty_classification = "known_trajectory_family"
        if similarity < 0.40:
            novelty_classification = "novel_trajectory"
        elif similarity < 0.60:
            novelty_classification = "weak_match"

        family_summary = self._clusterer.describe_family(family.family_id)
        progression = list(family_summary.get("next_path_distribution") or [])

        payload = {
            "status": "ready",
            "current_trajectory_family": family.family_id,
            "current_trajectory_archetype": family.family_id,
            "current_trajectory_path_family": family.label,
            "trajectory_family_label": family.label,
            "family_similarity": round(float(similarity), 4),
            "trajectory_similarity": round(float(similarity), 4),
            "nearest_trajectory_families": nearest,
            "nearest_trajectory_archetypes": nearest,
            "historical_escalation_frequency": round(float(family.escalation_events / max(1, family.support)), 4),
            "historical_phase_shift_frequency": round(float(family.phase_shift_events / max(1, family.support)), 4),
            "escalation_frequency_by_family": {
                family.family_id: round(float(family.escalation_events / max(1, family.support)), 4)
            },
            "support": int(family.support),
            "novelty_score": round(float(novelty_score), 4),
            "novelty_classification": novelty_classification,
            "matched_historical_progressions": [
                {
                    "path_family": str(item.get("path_family", "unknown")),
                    "support": int(item.get("support", 0)),
                    "frequency": float(item.get("frequency", 0.0)),
                    "co_observed_mechanisms": [m.get("mechanism") for m in family_summary.get("co_observed_mechanisms", [])],
                }
                for item in progression[:3]
            ],
            "representative_segment": family_summary.get("representative_segment", []),
            "trajectory_families": {
                "current": family_summary,
                "nearest": nearest,
            },
            "distance_inspection": family.last_alignment,
        }
        return payload
