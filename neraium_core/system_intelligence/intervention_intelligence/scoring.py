from __future__ import annotations

from typing import Any

from ..intervention_memory.memory import InterventionMemoryStore


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


class InterventionEffectivenessScorer:
    """Empirical structural effectiveness estimator (non-causal, advisory only)."""

    def __init__(self, memory: InterventionMemoryStore) -> None:
        self.memory = memory

    def score_candidate(
        self,
        *,
        context: dict[str, Any],
        intervention_type: str,
        intervention_target: str,
        model_projection: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        matches = self.memory.query(
            context=context,
            intervention_type=intervention_type,
            intervention_target=intervention_target,
            min_match=0.2,
        )
        support = len(matches)
        novelty = float(context.get("novelty_score", 0.5))
        avg_match = sum(float(x["match_quality"]) for x in matches) / max(1, support)

        escalation_reduction = 0.0
        reversibility = 0.0
        distance = 0.0
        worsening = 0.0
        if support > 0:
            weights = [float(item["match_quality"]) for item in matches]
            weight_total = sum(weights) or 1.0
            for item, w in zip(matches, weights):
                delta = item["record"]["outcome_delta"]
                escalation_reduction += w * float(delta.get("escalation_reduction", 0.0))
                reversibility += w * float(delta.get("reversibility_improvement", 0.0))
                distance += w * float(delta.get("distance_from_critical_improvement", 0.0))
                worsening += w * float(delta.get("escalation_worsening", 0.0))
            escalation_reduction /= weight_total
            reversibility /= weight_total
            distance /= weight_total
            worsening /= weight_total

        model_proj = model_projection or {}
        model_reduction = _clip01(float(model_proj.get("expected_escalation_reduction", 0.0)))

        support_factor = _clip01(support / 6.0)
        novelty_penalty = _clip01(0.42 * novelty + (1.0 - avg_match) * 0.42)
        uncertainty = _clip01(1.0 - (0.55 * support_factor + 0.45 * avg_match))
        memory_effect_weight = _clip01(0.25 + 0.55 * support_factor * avg_match)
        helpful_signal = _clip01(0.55 * escalation_reduction + 0.25 * reversibility + 0.20 * distance)
        harmful_signal_strength = _clip01(max(worsening, worsening - escalation_reduction + 0.04))

        expected_improvement = _clip01(
            memory_effect_weight * helpful_signal
            + (1.0 - memory_effect_weight) * model_reduction
            - 0.55 * harmful_signal_strength
        )
        raw_effectiveness = _clip01(expected_improvement * (1.0 - 0.48 * novelty_penalty) * (1.0 - 0.42 * uncertainty))

        harmful_signal = harmful_signal_strength > max(0.04, helpful_signal * 0.8)
        sparse_support = support < 2
        safety_gate = 1.0
        if sparse_support:
            safety_gate *= 0.80
        if harmful_signal:
            safety_gate *= 0.45
        effectiveness = _clip01(raw_effectiveness * safety_gate)

        return {
            "intervention_effectiveness": round(effectiveness, 4),
            "expected_trajectory_improvement": {
                "escalation_reduction": round(escalation_reduction, 4),
                "reversibility_improvement": round(reversibility, 4),
                "distance_from_critical_improvement": round(distance, 4),
                "model_projected_escalation_reduction": round(model_reduction, 4),
            },
            "support": support,
            "uncertainty": round(uncertainty, 4),
            "novelty_penalty": round(novelty_penalty, 4),
            "context_match_quality": round(avg_match, 4),
            "worsening_signal": round(worsening, 4),
            "harmful_signal_strength": round(harmful_signal_strength, 4),
            "support_quality": "strong" if support >= 4 else "moderate" if support >= 2 else "weak",
            "memory_effect_weight": round(memory_effect_weight, 4),
            "assumption_boundary": "Empirical structural effectiveness estimate from historical/contextual similarity and model projections; not a proven causal effect.",
            "safety_flags": {
                "sparse_support": sparse_support,
                "harmful_signal": harmful_signal,
            },
            "matched_records": matches[:5],
        }
