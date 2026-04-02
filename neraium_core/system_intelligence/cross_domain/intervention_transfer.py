from __future__ import annotations

from collections import defaultdict
from typing import Any


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class CrossDomainInterventionTransfer:
    """Experimental intervention transfer tracker with epistemic warnings."""

    def __init__(self) -> None:
        self._records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def update(
        self,
        *,
        domain: str,
        intervention_info: dict[str, Any],
        context_similarity: float,
    ) -> dict[str, Any]:
        ranked = list((intervention_info.get("recommendation") or {}).get("ranked_interventions") or [])
        if not ranked:
            return {
                "transferable_interventions": [],
                "domain_specific_interventions": [],
                "failed_transfers": [],
            }
        best = ranked[0]
        name = str(best.get("intervention_type") or best.get("name") or "unknown")
        historical = ((best.get("evidence_sources") or {}).get("historical_evidence") or {})
        effect = float(historical.get("intervention_effectiveness", 0.0))
        uncertainty = float(historical.get("uncertainty", 0.8))
        self._records[name].append(
            {
                "domain": str(domain or "unknown"),
                "effect": effect,
                "uncertainty": uncertainty,
                "context_similarity": float(context_similarity),
            }
        )

        transferable, domain_specific, failed = [], [], []
        for intervention, rows in self._records.items():
            domains = {r["domain"] for r in rows}
            avg_effect = sum(r["effect"] for r in rows) / len(rows)
            effect_var = sum((r["effect"] - avg_effect) ** 2 for r in rows) / len(rows)
            avg_similarity = sum(r["context_similarity"] for r in rows) / len(rows)
            transferability = _clamp(0.5 * avg_similarity + 0.35 * max(0.0, avg_effect) + 0.15 * (1.0 - min(1.0, effect_var)))
            payload = {
                "intervention": intervention,
                "transferability_score": round(float(transferability), 6),
                "domain_dependency": round(float(1.0 - min(1.0, len(domains) / 4.0)), 6),
                "generalization_warnings": [],
            }
            if len(domains) >= 2 and transferability >= 0.58:
                transferable.append(payload)
            elif len(domains) <= 1:
                payload["generalization_warnings"].append("single-domain evidence only")
                domain_specific.append(payload)
            else:
                payload["generalization_warnings"].append("cross-domain inconsistency")
                failed.append(payload)
        return {
            "transferable_interventions": transferable,
            "domain_specific_interventions": domain_specific,
            "failed_transfers": failed,
        }
