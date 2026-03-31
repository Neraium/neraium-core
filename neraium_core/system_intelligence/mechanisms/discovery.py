from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
from typing import Any


class MechanismDiscoveryLayer:
    """Discovers recurring interpretable mechanism candidates from graph + attribution signals."""

    def __init__(self) -> None:
        self._motif_counts: Counter[str] = Counter()
        self._motif_hits: Counter[str] = Counter()

    def update(self, *, top_relationships: list[dict[str, Any]], subsystem_impact: dict[str, float], escalating: bool) -> dict[str, Any]:
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
            if escalating:
                self._motif_hits[key] += 1
            recurrence = float(self._motif_counts[key])
            predictive = float(self._motif_hits[key] / max(1, self._motif_counts[key]))
            score = 0.45 * min(1.0, recurrence / 10.0) + 0.30 * strength + 0.25 * predictive
            ranked.append(
                {
                    "mechanism": key,
                    "evidence": {
                        "recurrence": int(self._motif_counts[key]),
                        "strength": round(float(strength), 4),
                        "predictive_value": round(float(predictive), 4),
                    },
                    "candidate_score": round(float(score), 4),
                    "classification": "candidate_mechanism",
                }
            )
        ranked = sorted(ranked, key=lambda item: item["candidate_score"], reverse=True)[:5]

        return {
            "status": "ready",
            "method": "dynamic motif + subsystem decoupling recurrence",
            "mechanism_candidates": ranked,
            "disclaimer": "Mechanisms are ranked structural hypotheses from recurring motifs, not proven causal mechanisms.",
        }
