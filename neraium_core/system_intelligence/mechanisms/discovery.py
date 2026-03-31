from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any


class MechanismDiscoveryLayer:
    """Discovers recurring interpretable mechanism candidates from graph + attribution signals."""

    def __init__(self) -> None:
        self._motif_counts: Counter[str] = Counter()
        self._motif_hits: Counter[str] = Counter()
        self._motif_recovery_hits: Counter[str] = Counter()
        self._family_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._family_escalation_hits: dict[str, Counter[str]] = defaultdict(Counter)
        self._family_recovery_hits: dict[str, Counter[str]] = defaultdict(Counter)

    def update(
        self,
        *,
        top_relationships: list[dict[str, Any]],
        subsystem_impact: dict[str, float],
        escalating: bool,
        trajectory_family: str = "unknown",
        recovering: bool = False,
    ) -> dict[str, Any]:
        candidate_motifs: list[tuple[str, float]] = []

        rel_pairs = []
        for rel in top_relationships[:5]:
            a = str(rel.get("source", "")).strip()
            b = str(rel.get("target", "")).strip()
            if a and b:
                rel_pairs.append(tuple(sorted((a, b))))
        nodes = sorted({n for pair in rel_pairs for n in pair})
        for triad in combinations(nodes[:6], 3):
            key = f"triad_weakening:{triad[0]}|{triad[1]}|{triad[2]}"
            strength = 0.3 + 0.1 * len([p for p in rel_pairs if p[0] in triad and p[1] in triad])
            candidate_motifs.append((key, min(1.0, strength)))

        numeric_items: list[tuple[str, float]] = []
        if subsystem_impact:
            for key, value in subsystem_impact.items():
                if isinstance(value, (int, float)):
                    numeric_items.append((str(key), float(value)))
                elif isinstance(value, dict):
                    score = value.get("score")
                    if isinstance(score, (int, float)):
                        numeric_items.append((str(key), float(score)))
        if numeric_items:
            items = sorted(numeric_items, key=lambda kv: kv[1], reverse=True)
            if len(items) >= 2:
                key = f"cluster_decoupling:{items[0][0]}->{items[1][0]}"
                candidate_motifs.append((key, min(1.0, 0.5 + 0.5 * items[0][1])))

        ranked = []
        for key, strength in candidate_motifs[:8]:
            self._motif_counts[key] += 1
            self._family_counts[trajectory_family][key] += 1
            if escalating:
                self._motif_hits[key] += 1
                self._family_escalation_hits[trajectory_family][key] += 1
            if recovering:
                self._motif_recovery_hits[key] += 1
                self._family_recovery_hits[trajectory_family][key] += 1

            recurrence = float(self._motif_counts[key])
            predictive = float(self._motif_hits[key] / max(1, self._motif_counts[key]))
            recovery_assoc = float(self._motif_recovery_hits[key] / max(1, self._motif_counts[key]))
            conditioned_predictive = float(
                self._family_escalation_hits[trajectory_family][key] / max(1, self._family_counts[trajectory_family][key])
            )
            conditioned_recovery = float(
                self._family_recovery_hits[trajectory_family][key] / max(1, self._family_counts[trajectory_family][key])
            )
            delta = conditioned_predictive - conditioned_recovery
            score = 0.35 * min(1.0, recurrence / 10.0) + 0.25 * strength + 0.25 * predictive + 0.15 * max(0.0, delta)

            association = "common_but_weak"
            if delta >= 0.18 and self._family_counts[trajectory_family][key] >= 3:
                association = "escalation_correlated"
            elif conditioned_recovery >= 0.55 and self._family_counts[trajectory_family][key] >= 3:
                association = "recovery_associated"

            ranked.append(
                {
                    "mechanism": key,
                    "evidence": {
                        "recurrence": int(self._motif_counts[key]),
                        "strength": round(float(strength), 4),
                        "predictive_value": round(float(predictive), 4),
                        "recovery_association": round(float(recovery_assoc), 4),
                    },
                    "family_conditioned_predictive_value": round(float(conditioned_predictive), 4),
                    "family_conditioned_recovery_value": round(float(conditioned_recovery), 4),
                    "association": association,
                    "candidate_score": round(float(score), 4),
                    "classification": "candidate_mechanism",
                }
            )
        ranked = sorted(ranked, key=lambda item: item["candidate_score"], reverse=True)[:5]

        evidence_by_archetype = {
            item["mechanism"]: {
                "trajectory_family": trajectory_family,
                "support": int(self._family_counts[trajectory_family][item["mechanism"]]),
                "escalation_rate": round(
                    float(
                        self._family_escalation_hits[trajectory_family][item["mechanism"]]
                        / max(1, self._family_counts[trajectory_family][item["mechanism"]])
                    ),
                    4,
                ),
                "recovery_rate": round(
                    float(
                        self._family_recovery_hits[trajectory_family][item["mechanism"]]
                        / max(1, self._family_counts[trajectory_family][item["mechanism"]])
                    ),
                    4,
                ),
            }
            for item in ranked
        }

        return {
            "status": "ready",
            "method": "dynamic motif + subsystem decoupling recurrence",
            "mechanism_candidates": ranked,
            "mechanism_trajectory_associations": [
                {
                    "mechanism": item["mechanism"],
                    "trajectory_family": trajectory_family,
                    "family_conditioned_predictive_value": item["family_conditioned_predictive_value"],
                    "association": item["association"],
                }
                for item in ranked
            ],
            "evidence_by_archetype": evidence_by_archetype,
            "disclaimer": "Mechanisms are ranked structural hypotheses from recurring motifs, not proven causal mechanisms.",
        }
