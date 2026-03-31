from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
            worsening_signal = float(score.get("worsening_signal", 0.0))

            consistent_with_laws = 1.0 if any(term in name for term in law_candidates) else 0.6
            family_consistency = 1.0 if family in {"escalating", "reversible", "drift", "stable"} else 0.7

            composite = (
                0.44 * expected
                + 0.14 * min(1.0, support / 6.0)
                + 0.10 * context_match
                + 0.12 * consistent_with_laws
                + 0.08 * family_consistency
                - 0.14 * uncertainty
                - 0.10 * novelty
                - 0.12 * worsening_signal
            )
            confidence = _clip01(composite)
            if support < 2:
                confidence *= 0.82
            if worsening_signal > 0.05:
                confidence *= 0.7

            ranked.append(
                {
                    "name": name,
                    "intervention_type": cand["intervention_type"],
                    "intervention_target": cand["intervention_target"],
                    "rank_score": round(confidence, 4),
                    "confidence": round(confidence, 4),
                    "rationale": "Advisory ranking from bounded evidence and projection signals; not proof of causal effect.",
                    "ranking_factors": {
                        "expected_effectiveness": round(expected, 4),
                        "support": int(support),
                        "uncertainty": round(uncertainty, 4),
                        "novelty_penalty": round(novelty, 4),
                        "worsening_signal": round(worsening_signal, 4),
                        "context_match": round(context_match, 4),
                    },
                    "evidence_sources": {
                        "model_based_projection": cand.get("model_projection") or {},
                        "historical_evidence": score,
                    },
                }
            )

        ranked.sort(key=lambda x: float(x["rank_score"]), reverse=True)
        best = ranked[0] if ranked else None

        for idx, item in enumerate(ranked):
            lower_reasons: list[str] = []
            for other in ranked[:idx]:
                if float(other["rank_score"]) <= float(item["rank_score"]):
                    continue
                lower_reasons.append(
                    f"Lower than {other['name']} due to weaker support/effectiveness or higher uncertainty/novelty."
                )
            item["why_ranked_lower"] = lower_reasons[:2]

        best_conf = float(best["confidence"]) if best else 0.0
        return {
            "ranked_interventions": ranked,
            "best_intervention": best,
            "confidence": round(best_conf, 4),
            "recommendation_confidence": round(best_conf, 4),
            "decision_trace": {
                "confidence_contributions": {
                    "historical_evidence": "primary when support and context match are strong",
                    "counterfactual_projection": "secondary projection input",
                    "law_and_trajectory_consistency": "consistency signal",
                    "uncertainty_and_novelty_penalties": "always applied to keep recommendations conservative",
                },
                "assumptions_remaining": [
                    "Historical similarity may omit hidden confounders.",
                    "Counterfactual projections are approximate structural simulations.",
                    "Recommendations are advisory and require operator judgement.",
                ],
            },
        }
