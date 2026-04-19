from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class LawEvidence:
    support: int = 0
    escalation_hits: int = 0
    recovery_hits: int = 0
    assets: set[str] | None = None

    def __post_init__(self) -> None:
        if self.assets is None:
            self.assets = set()


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
        counterfactual_info: dict[str, Any] | None = None,
        asset_id: str = "unknown",
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
            ev.assets.add(asset_id)
            if escalating:
                ev.escalation_hits += 1
            if recovering:
                ev.recovery_hits += 1

            escalation_rate = float(ev.escalation_hits / max(1, ev.support))
            recovery_rate = float(ev.recovery_hits / max(1, ev.support))
            support_strength = min(1.0, ev.support / max(1.0, float(self.robust_support)))
            effect = escalation_rate - recovery_rate
            robustness = 0.55 * support_strength + 0.45 * abs(escalation_rate - 0.5) * 2.0
            consistency = min(1.0, len(ev.assets) / 6.0)
            trajectory_specificity = min(1.0, 0.45 + 0.55 * float(item.get("family_conditioned_predictive_value", 0.0)))
            cf_consistency = 0.0
            if isinstance(counterfactual_info, dict):
                scenarios = list(counterfactual_info.get("scenario_rankings") or [])
                if scenarios:
                    best_delta = max(float(s.get("risk_delta", 0.0)) for s in scenarios[:2])
                    cf_consistency = min(1.0, max(0.0, best_delta))
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
                    "mechanism": mechanism,
                    "estimated_effect": round(float(effect), 4),
                    "robustness": round(float(max(0.0, min(1.0, robustness))), 4),
                    "consistency_across_assets_runs": round(float(consistency), 4),
                    "trajectory_family_support": round(float(trajectory_specificity), 4),
                    "counterfactual_consistency": round(float(cf_consistency), 4),
                    "confidence": round(float(max(0.0, min(1.0, 0.35 * support_strength + 0.35 * abs(effect) + 0.15 * consistency + 0.15 * trajectory_specificity))), 4),
                    "classification": classification,
                    "applicability": {
                        "archetypes": [archetype],
                        "regimes": [str(transition.get("regime", "unknown"))],
                        "trajectory_family": path_family,
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
