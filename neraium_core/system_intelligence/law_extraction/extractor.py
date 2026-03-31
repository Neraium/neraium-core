from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class LawEvidence:
    support: int = 0
    escalation_hits: int = 0
    recovery_hits: int = 0


class StructuralLawExtractor:
    """Extracts interpretable, bounded structural law candidates from recurring evidence."""

    def __init__(self, *, min_support: int = 4, robust_support: int = 10) -> None:
        self.min_support = int(min_support)
        self.robust_support = int(robust_support)
        self._rules: dict[str, LawEvidence] = defaultdict(LawEvidence)

    def update(
        self,
        *,
        trajectory_info: dict[str, Any],
        mechanism_info: dict[str, Any],
        transition: dict[str, Any],
    ) -> dict[str, Any]:
        if trajectory_info.get("status") != "ready":
            return {"status": "warming", "law_candidates": []}

        archetype = str(trajectory_info.get("current_trajectory_archetype") or "unknown")
        path_family = str(trajectory_info.get("current_trajectory_path_family") or "unknown")
        transition_path = str(transition.get("transition_path") or "unknown")
        escalating = float(transition.get("escalation_probability", 0.0)) >= 0.6 or transition_path == "escalating"
        recovering = transition_path == "reversible"

        candidates = []
        for item in list(mechanism_info.get("mechanism_candidates") or [])[:4]:
            mechanism = str(item.get("mechanism", "unknown"))
            key = f"{archetype}|{path_family}|{mechanism}"
            ev = self._rules[key]
            ev.support += 1
            if escalating:
                ev.escalation_hits += 1
            if recovering:
                ev.recovery_hits += 1

            escalation_rate = float(ev.escalation_hits / max(1, ev.support))
            recovery_rate = float(ev.recovery_hits / max(1, ev.support))
            support_strength = min(1.0, ev.support / max(1.0, float(self.robust_support)))
            effect = max(0.0, escalation_rate - recovery_rate)
            robustness = 0.55 * support_strength + 0.45 * abs(escalation_rate - 0.5) * 2.0
            if ev.support >= self.min_support and effect >= 0.18:
                classification = "candidate_structural_law"
            elif ev.support >= self.min_support and effect >= 0.08:
                classification = "weak_pattern"
            else:
                classification = "unsupported_hypothesis"

            candidates.append(
                {
                    "law_id": f"law::{key}",
                    "condition": f"Trajectory family '{path_family}' with mechanism '{mechanism}' in archetype '{archetype}'.",
                    "outcome": "Escalation tendency increases in subsequent transition phase.",
                    "support": int(ev.support),
                    "estimated_effect": round(float(effect), 4),
                    "robustness": round(float(max(0.0, min(1.0, robustness))), 4),
                    "confidence": round(float(max(0.0, min(1.0, 0.5 * support_strength + 0.5 * effect))), 4),
                    "classification": classification,
                    "applicability": {
                        "archetypes": [archetype],
                        "regimes": [str(transition.get("regime", "unknown"))],
                        "notes": "Candidate regularity from recurring co-occurrence; not a scientific proof.",
                    },
                    "evidence_snapshot": {
                        "escalation_rate": round(escalation_rate, 4),
                        "recovery_rate": round(recovery_rate, 4),
                    },
                }
            )

        candidates.sort(key=lambda row: (row["classification"] == "candidate_structural_law", row["robustness"], row["support"]), reverse=True)
        return {"status": "ready", "law_candidates": candidates[:6]}
