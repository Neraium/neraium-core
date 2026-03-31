from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neraium_core.system_intelligence.trajectory_alignment import AlignmentResult, DTWAligner


@dataclass
class TrajectoryFamilyRecord:
    family_id: str
    label: str
    medoid: np.ndarray
    support: int = 0
    escalation_events: int = 0
    phase_shift_events: int = 0
    critical_hits: int = 0
    transition_counts: dict[str, int] = field(default_factory=dict)
    mechanism_counts: dict[str, int] = field(default_factory=dict)
    steps_to_critical: list[int] = field(default_factory=list)
    last_alignment: dict[str, float] = field(default_factory=dict)


class TrajectoryFamilyClusterer:
    """Online medoid-style trajectory family clustering backed by DTW alignment."""

    def __init__(self, *, max_families: int = 18, min_similarity_for_match: float = 0.56) -> None:
        self.max_families = int(max_families)
        self.min_similarity_for_match = float(min_similarity_for_match)
        self.aligner = DTWAligner(window_ratio=0.25, prefilter_weight=0.25)
        self._families: dict[str, TrajectoryFamilyRecord] = {}

    def assign(
        self,
        *,
        segment: np.ndarray,
        inferred_label: str,
        escalating: bool,
        in_critical_region: bool,
        phase_shift: bool,
        mechanism_names: list[str],
    ) -> tuple[TrajectoryFamilyRecord, list[dict[str, object]], float]:
        match_id, best_similarity, nearest = self._nearest(segment)
        if match_id is None:
            family = self._create_family(segment=segment, label=inferred_label)
            best_similarity = 1.0 if not self._families else max(best_similarity, 0.0)
        else:
            family = self._families[match_id]

        self._update_family(
            family=family,
            segment=segment,
            inferred_label=inferred_label,
            escalating=escalating,
            in_critical_region=in_critical_region,
            phase_shift=phase_shift,
            mechanism_names=mechanism_names,
        )
        return family, nearest, float(best_similarity)

    def _create_family(self, *, segment: np.ndarray, label: str) -> TrajectoryFamilyRecord:
        if len(self._families) >= self.max_families:
            replace = min(
                self._families.values(),
                key=lambda f: (f.support, f.escalation_events + f.phase_shift_events),
            )
            del self._families[replace.family_id]

        family_id = f"trajectory_family_{len(self._families) + 1}"
        family = TrajectoryFamilyRecord(family_id=family_id, label=label, medoid=segment.copy())
        self._families[family_id] = family
        return family

    def _nearest(self, segment: np.ndarray) -> tuple[str | None, float, list[dict[str, object]]]:
        if not self._families:
            return None, 0.0, []

        scored: list[tuple[str, AlignmentResult]] = []
        for family_id, family in self._families.items():
            alignment = self.aligner.compare(segment, family.medoid)
            scored.append((family_id, alignment))
        scored.sort(key=lambda row: row[1].similarity, reverse=True)

        nearest: list[dict[str, object]] = []
        for family_id, alignment in scored[:3]:
            f = self._families[family_id]
            nearest.append(
                {
                    "trajectory_family": family_id,
                    "family_label": f.label,
                    "family_similarity": round(float(alignment.similarity), 4),
                    "support": int(f.support),
                    "historical_escalation_frequency": round(float(f.escalation_events / max(1, f.support)), 4),
                    "escalation_frequency": round(float(f.escalation_events / max(1, f.support)), 4),
                    "alignment": {
                        "normalized_distance": round(float(alignment.normalized_distance), 5),
                        "path_length": int(alignment.path_length),
                        "diagonal_fraction": round(float(alignment.diagonal_fraction), 5),
                    },
                }
            )

        best_id, best_alignment = scored[0]
        if best_alignment.similarity < self.min_similarity_for_match:
            return None, float(best_alignment.similarity), nearest
        return best_id, float(best_alignment.similarity), nearest

    def _update_family(
        self,
        *,
        family: TrajectoryFamilyRecord,
        segment: np.ndarray,
        inferred_label: str,
        escalating: bool,
        in_critical_region: bool,
        phase_shift: bool,
        mechanism_names: list[str],
    ) -> None:
        family.support += 1
        alignment = self.aligner.compare(segment, family.medoid)
        family.last_alignment = {
            "similarity": round(float(alignment.similarity), 6),
            "normalized_distance": round(float(alignment.normalized_distance), 6),
            "path_length": int(alignment.path_length),
        }
        # online medoid-ish update via running average; keeps representation interpretable
        lr = 1.0 / float(family.support)
        family.medoid = (1.0 - lr) * family.medoid + lr * segment

        if family.support <= 3:
            family.label = inferred_label
        elif family.support % 7 == 0:
            # keep label adaptive but stable over time
            if inferred_label != family.label and (family.escalation_events / max(1, family.support)) > 0.65:
                family.label = "escalating"

        if escalating:
            family.escalation_events += 1
        if phase_shift:
            family.phase_shift_events += 1
        if in_critical_region:
            family.critical_hits += 1
            steps_to_critical = max(1, int(np.ceil(np.linalg.norm(segment[-1]) * 1.8)))
            family.steps_to_critical.append(steps_to_critical)

        family.transition_counts[inferred_label] = family.transition_counts.get(inferred_label, 0) + 1
        for name in mechanism_names[:3]:
            family.mechanism_counts[name] = family.mechanism_counts.get(name, 0) + 1

    def describe_family(self, family_id: str) -> dict[str, object]:
        family = self._families[family_id]
        top_next = sorted(family.transition_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_mechanisms = sorted(family.mechanism_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return {
            "trajectory_family": family_id,
            "family_label": family.label,
            "support": int(family.support),
            "escalation_frequency": round(float(family.escalation_events / max(1, family.support)), 4),
            "phase_shift_frequency": round(float(family.phase_shift_events / max(1, family.support)), 4),
            "critical_frequency": round(float(family.critical_hits / max(1, family.support)), 4),
            "representative_segment": [[round(float(v), 5) for v in row] for row in family.medoid[-5:]],
            "next_path_distribution": [
                {"path_family": key, "support": int(cnt), "frequency": round(float(cnt / max(1, family.support)), 4)}
                for key, cnt in top_next
            ],
            "co_observed_mechanisms": [{"mechanism": k, "support": int(v)} for k, v in top_mechanisms],
        }
