from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


@dataclass
class DomainEvidence:
    support: int = 0
    escalation_hits: int = 0


class UniversalStructuralLawEngine:
    """Experimental cross-domain law candidate tracker. Never treated as production truth."""

    def __init__(self, *, universal_min_domains: int = 3, universal_min_support: int = 14) -> None:
        self.universal_min_domains = int(universal_min_domains)
        self.universal_min_support = int(universal_min_support)
        self._rules: dict[str, dict[str, DomainEvidence]] = defaultdict(lambda: defaultdict(DomainEvidence))

    def update(
        self,
        *,
        domain: str,
        trajectory_archetype_id: str,
        mechanism: str,
        escalating: bool,
    ) -> dict[str, Any]:
        dom = str(domain or "unknown")
        key = f"{trajectory_archetype_id}|{mechanism}"
        evidence = self._rules[key][dom]
        evidence.support += 1
        if escalating:
            evidence.escalation_hits += 1

        by_domain = self._rules[key]
        support = sum(v.support for v in by_domain.values())
        rates = {k: (v.escalation_hits / max(1, v.support)) for k, v in by_domain.items()}
        if rates:
            mean_rate = sum(rates.values()) / len(rates)
            variance = sum((r - mean_rate) ** 2 for r in rates.values()) / len(rates)
        else:
            mean_rate = 0.0
            variance = 1.0
        effect_consistency = _clamp(1.0 - (variance**0.5))
        if len(rates) >= self.universal_min_domains and support >= self.universal_min_support and effect_consistency >= 0.72:
            classification = "candidate_universal_structural_law"
        elif len(rates) >= 2:
            classification = "multi_system_pattern"
        else:
            classification = "local_law"
        return {
            "law_key": key,
            "classification": classification,
            "cross_domain_support": int(support),
            "cross_domain_effect_consistency": round(float(effect_consistency), 6),
            "domain_variance": round(float(variance), 6),
            "domain_rates": {k: round(float(v), 6) for k, v in rates.items()},
            "epistemic_note": "Experimental invariant candidate. Not a confirmed universal law.",
        }
