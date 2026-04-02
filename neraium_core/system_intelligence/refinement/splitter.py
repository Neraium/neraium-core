from __future__ import annotations

from collections import defaultdict
from typing import Any


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


class LawRefinementEngine:
    """Refines (splits) contradicted laws into narrower conditional variants."""

    def refine(self, *, law: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        if not evidence:
            return {"refined_law_variants": [], "reduced_contradiction_rate": None}

        base_rate = sum(1 for row in evidence if row.get("contradiction")) / max(1, len(evidence))
        by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evidence:
            ctx = f"{row.get('domain')}|{row.get('regime')}|{row.get('trajectory_family')}"
            by_context[ctx].append(row)

        variants = []
        best_rate = base_rate
        for ctx, rows in by_context.items():
            support = len(rows)
            contradiction_rate = sum(1 for r in rows if r.get("contradiction")) / max(1, support)
            if support < 2:
                continue
            if contradiction_rate >= base_rate:
                continue
            domain, regime, family = ctx.split("|", 2)
            best_rate = min(best_rate, contradiction_rate)
            variants.append(
                {
                    "variant_law_id": f"{law.get('law_id', 'law::unknown')}::variant::{len(variants)+1}",
                    "base_law_id": law.get("law_id"),
                    "condition": {
                        "domain": domain,
                        "regime": regime,
                        "trajectory_family": family,
                    },
                    "support": support,
                    "contradiction_rate": round(float(_clip01(contradiction_rate)), 6),
                    "scope_note": "Narrowed applicability to reduce false generalization.",
                }
            )

        return {
            "refined_law_variants": variants,
            "reduced_contradiction_rate": round(float(max(0.0, base_rate - best_rate)), 6),
            "narrower_applicability_scope": bool(variants),
        }
