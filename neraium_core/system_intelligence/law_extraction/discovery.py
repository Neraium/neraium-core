from __future__ import annotations

from collections import defaultdict


class StructuralLawDiscoveryLayer:
    """Extracts candidate structural laws from trajectory + mechanism recurrence."""

    def __init__(self) -> None:
        self._counts = defaultdict(int)
        self._hits = defaultdict(int)

    def update(self, *, trajectory_info: dict, mechanisms: dict, transition, escalating: bool) -> dict:
        laws = []
        traj = trajectory_info.get("current_trajectory_archetype", "unknown")
        for mech in (mechanisms.get("mechanism_candidates") or [])[:3]:
            key = f"{traj}|{mech.get('mechanism')}"
            self._counts[key] += 1
            if escalating:
                self._hits[key] += 1
            count = self._counts[key]
            effect = self._hits[key] / max(1, count)
            score = effect * (1.0 + count / 5.0)
            laws.append({
                "law_id": key,
                "condition": f"trajectory={traj} & mechanism={mech.get('mechanism')}",
                "outcome": "escalation_likelihood",
                "support": count,
                "estimated_effect": round(effect, 4),
                "robustness": round(min(1.0, count / 10.0), 4),
                "classification": "candidate_structural_law",
                "score": round(score, 4),
            })
        laws = sorted(laws, key=lambda x: x["score"], reverse=True)[:3]
        return {"law_candidates": laws}
