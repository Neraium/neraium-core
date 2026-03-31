from __future__ import annotations

from typing import Any


class InterventionRecommendationRanker:
    """Ranks interventions for operator support with bounded, explainable scoring."""

    def rank(
        self,
        *,
        candidates: list[dict[str, Any]],
        scored: dict[str, dict[str, Any]],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        ranked: list[dict[str, Any]] = []
        law_candidates = set(context.get("law_candidates") or [])
        family = str(context.get("trajectory_family", "unknown"))

        for cand in candidates:
            name = str(cand["name"])
            score = scored[name]
            expected = float(score["intervention_effectiveness"])
            support = float(score["support"])
            uncertainty = float(score["uncertainty"])
            novelty = float(score["novelty_penalty"])
            context_match = float(score["context_match_quality"])

            consistent_with_laws = 1.0 if any(term in name for term in law_candidates) else 0.6
            family_consistency = 1.0 if family in {"escalating", "reversible", "drift", "stable"} else 0.7

            composite = (
                0.40 * expected
                + 0.15 * min(1.0, support / 6.0)
                + 0.10 * context_match
                + 0.15 * consistent_with_laws
                + 0.10 * family_consistency
                - 0.10 * uncertainty
                - 0.10 * novelty
            )
            confidence = max(0.0, min(1.0, composite))

            ranked.append(
                {
                    "name": name,
                    "intervention_type": cand["intervention_type"],
                    "intervention_target": cand["intervention_target"],
                    "rank_score": round(confidence, 4),
                    "confidence": round(confidence, 4),
                    "rationale": (
                        "Advisory ranking based on expected structural benefit, support, uncertainty, novelty, and consistency with trajectory/law evidence."
                    ),
                    "evidence_sources": {
                        "model_based_projection": cand.get("model_projection") or {},
                        "historical_effectiveness": score,
                    },
                }
            )

        ranked.sort(key=lambda x: float(x["rank_score"]), reverse=True)
        best = ranked[0] if ranked else None

        for idx, item in enumerate(ranked):
            lower = []
            for other in ranked[:idx]:
                if float(other["rank_score"]) > float(item["rank_score"]):
                    lower.append(f"Below {other['name']} due to lower support/effectiveness or higher novelty/uncertainty")
            item["why_ranked_lower"] = lower[:2]

        return {
            "ranked_interventions": ranked,
            "best_intervention": best,
            "confidence": round(float(best["confidence"]) if best else 0.0, 4),
            "decision_trace": {
                "confidence_contributions": {
                    "historical_memory": "primary when support and context match are strong",
                    "counterfactual_projection": "secondary projection input",
                    "law_and_trajectory_consistency": "consistency multiplier",
                },
                "assumptions_remaining": [
                    "Historical similarity may omit hidden confounders.",
                    "Counterfactual projections are approximate structural simulations.",
                    "Recommendations are advisory and require operator judgement.",
                ],
            },
        }
