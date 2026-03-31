from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from neraium_core.system_intelligence.trajectory_archetypes.archetypes import (
    TrajectoryArchetype,
    TrajectorySegment,
    TrajectorySegmentEncoder,
    dtw_distance,
)


class CrossSystemTrajectoryMemory:
    """Cross-asset memory of representative trajectory segments and outcome statistics."""

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
        self._archetypes: dict[str, TrajectoryArchetype] = {}
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
                "nearest_trajectory_archetypes": [],
                "trajectory_similarity": 0.0,
                "novelty_score": 1.0,
                "novelty_classification": "insufficient_history",
            }

        current_id, similarity, nearest = self._match_archetype(segment)
        if current_id is None:
            current_id = self._create_archetype(segment)
            similarity = 1.0

        archetype = self._archetypes[current_id]
        self._update_archetype_stats(
            archetype=archetype,
            segment=segment,
            escalating=bool(escalating),
            in_critical_region=bool(in_critical_region),
            phase_shift=bool(phase_shift),
        )

        novelty_score = max(0.0, min(1.0, 1.0 - similarity))
        novelty_classification = "known_trajectory_family"
        if similarity < 0.40:
            novelty_classification = "novel_trajectory"
        elif similarity < 0.60:
            novelty_classification = "weak_match"

        return {
            "status": "ready",
            "current_trajectory_archetype": current_id,
            "current_trajectory_path_family": segment.path_family,
            "trajectory_similarity": round(float(similarity), 4),
            "nearest_trajectory_archetypes": nearest,
            "historical_escalation_frequency": round(float(archetype.escalation_events / max(1, archetype.support)), 4),
            "historical_phase_shift_frequency": round(float(archetype.phase_shift_events / max(1, archetype.support)), 4),
            "support": int(archetype.support),
            "novelty_score": round(float(novelty_score), 4),
            "novelty_classification": novelty_classification,
            "matched_historical_progressions": self._build_progressions(current_id, mechanism_names=mechanism_names),
        }

    def _build_progressions(self, archetype_id: str, mechanism_names: list[str]) -> list[dict[str, object]]:
        archetype = self._archetypes[archetype_id]
        top_next = sorted(archetype.next_path_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        out = []
        for nxt, cnt in top_next:
            out.append(
                {
                    "path_family": nxt,
                    "support": int(cnt),
                    "frequency": round(float(cnt / max(1, archetype.support)), 4),
                    "co_observed_mechanisms": list(mechanism_names[:3]),
                }
            )
        return out

    def _create_archetype(self, segment: TrajectorySegment) -> str:
        archetype_id = f"trajectory_archetype_{len(self._archetypes) + 1}"
        if len(self._archetypes) >= self.max_archetypes:
            replace = sorted(self._archetypes.items(), key=lambda kv: kv[1].support)[0][0]
            archetype_id = replace
            del self._archetypes[replace]
        self._archetypes[archetype_id] = TrajectoryArchetype(archetype_id=archetype_id, prototype=segment.points.copy())
        return archetype_id

    def _match_archetype(self, segment: TrajectorySegment) -> tuple[str | None, float, list[dict[str, object]]]:
        if not self._archetypes:
            return None, 0.0, []
        scored = []
        feat_a = self.encoder.segment_feature_vector(segment.points)
        for archetype_id, archetype in self._archetypes.items():
            dtw = dtw_distance(segment.points, archetype.prototype)
            feat_b = self.encoder.segment_feature_vector(archetype.prototype)
            feat_d = float(np.linalg.norm(feat_a - feat_b) / (np.sqrt(feat_a.size) + 1e-6))
            combined = 0.68 * dtw + 0.32 * feat_d
            similarity = float(np.exp(-combined))
            scored.append((archetype_id, similarity, combined))

        scored.sort(key=lambda row: row[1], reverse=True)
        nearest = []
        for archetype_id, similarity, _ in scored[:3]:
            a = self._archetypes[archetype_id]
            nearest.append(
                {
                    "trajectory_archetype": archetype_id,
                    "similarity": round(float(similarity), 4),
                    "support": int(a.support),
                    "historical_escalation_frequency": round(float(a.escalation_events / max(1, a.support)), 4),
                    "historical_phase_shift_frequency": round(float(a.phase_shift_events / max(1, a.support)), 4),
                }
            )

        best_id, best_similarity, _ = scored[0]
        if best_similarity < self.min_similarity_for_match:
            return None, best_similarity, nearest
        return best_id, best_similarity, nearest

    def _update_archetype_stats(
        self,
        *,
        archetype: TrajectoryArchetype,
        segment: TrajectorySegment,
        escalating: bool,
        in_critical_region: bool,
        phase_shift: bool,
    ) -> None:
        archetype.support += 1
        lr = 1.0 / float(archetype.support)
        archetype.prototype = (1.0 - lr) * archetype.prototype + lr * segment.points
        if escalating or in_critical_region:
            archetype.escalation_events += 1
        if phase_shift:
            archetype.phase_shift_events += 1
        archetype.next_path_counts[segment.path_family] = archetype.next_path_counts.get(segment.path_family, 0) + 1

        if in_critical_region:
            steps_to_critical = max(1, int(np.ceil(np.linalg.norm(segment.points[-1]) * 2.0)))
            archetype.steps_to_critical.append(steps_to_critical)
