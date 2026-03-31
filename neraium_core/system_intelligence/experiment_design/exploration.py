from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def score_exploration_vs_exploitation(
    *,
    intervention: dict[str, Any],
    candidate_experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = (intervention.get("recommendation") or {}).get("ranked_interventions") or []
    enriched: list[dict[str, Any]] = []
    for row in ranked:
        hist = ((row.get("evidence_sources") or {}).get("historical_evidence") or {})
        improvement = float(hist.get("intervention_effectiveness", row.get("projected_effectiveness", 0.0)))
        uncertainty = float(hist.get("uncertainty", 0.5))
        learning_value = _clip01(0.6 * uncertainty + 0.4 * float((row.get("confidence") or 0.5) < 0.55))
        exploitation_score = _clip01(0.75 * max(0.0, improvement) + 0.25 * (1.0 - uncertainty))
        exploration_score = _clip01(0.65 * learning_value + 0.35 * uncertainty)
        enriched.append(
            {
                **row,
                "expected_outcome_improvement": round(improvement, 4),
                "expected_learning_value": round(learning_value, 4),
                "exploitation_score": round(exploitation_score, 4),
                "exploration_score": round(exploration_score, 4),
            }
        )

    exploit = sorted(enriched, key=lambda x: x.get("exploitation_score", 0.0), reverse=True)
    explore = sorted(enriched, key=lambda x: x.get("exploration_score", 0.0), reverse=True)
    top_experiment = candidate_experiments[0] if candidate_experiments else {}

    return {
        "exploit_recommendations": exploit[:3],
        "explore_recommendations": [
            {
                "name": top_experiment.get("name", "increase_observability"),
                "experiment_type": top_experiment.get("experiment_type", "measurement"),
                "expected_information_gain": top_experiment.get("expected_information_gain", 0.0),
                "discriminative_power": top_experiment.get("discriminative_power", 0.0),
            }
        ] + explore[:2],
        "exploration_vs_exploitation_tradeoff": {
            "recommended_mode": "explore" if top_experiment.get("expected_information_gain", 0.0) > 0.45 else "exploit",
            "exploration_pressure": round(float(top_experiment.get("expected_information_gain", 0.0)), 4),
            "exploitation_pressure": round(float(exploit[0].get("exploitation_score", 0.0)) if exploit else 0.0, 4),
            "advisory": True,
        },
    }
