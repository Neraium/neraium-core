from __future__ import annotations

from typing import Any


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def recommend_measurements(
    *,
    observation: dict[str, Any],
    trajectory: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    uncertainty: dict[str, Any],
) -> dict[str, Any]:
    subsystem_impact = dict(observation.get("subsystem_impact") or {})
    subsystem_scores: dict[str, float] = {}
    for key, value in subsystem_impact.items():
        try:
            subsystem_scores[str(key)] = float(value)
        except Exception:
            subsystem_scores[str(key)] = float(len(value)) if isinstance(value, (list, tuple, set)) else 0.0
    top_subsystem = max(subsystem_scores, key=subsystem_scores.get) if subsystem_scores else "subsystem_x"

    sensor_count = float(len(list(observation.get("sensor_readings") or [])) or len(list(observation.get("sensors") or [])))
    novelty = float(trajectory.get("novelty_score", 0.5))
    ambiguity = float((uncertainty.get("uncertainty_vector") or {}).get("structural_ambiguity", 0.5))
    observability_gap = _clip01(0.55 * (1.0 / (1.0 + sensor_count / 3.0)) + 0.25 * novelty + 0.2 * ambiguity)

    strong_competition = len(hypotheses) >= 2 and abs(float(hypotheses[0].get("likelihood", 0.0)) - float(hypotheses[1].get("likelihood", 0.0))) <= 0.15
    recommendations = [
        {
            "measurement": f"increase_sampling_frequency::{top_subsystem}",
            "rationale": "Higher temporal resolution can separate drift signatures from oscillatory signatures.",
            "expected_uncertainty_reduction": round(_clip01(0.4 + 0.35 * observability_gap), 4),
        },
        {
            "measurement": f"add_redundant_sensor::{top_subsystem}",
            "rationale": "Redundant sensing reduces hidden-variable ambiguity and calibration blind spots.",
            "expected_uncertainty_reduction": round(_clip01(0.35 + 0.3 * observability_gap), 4),
        },
    ]
    if strong_competition:
        recommendations.append(
            {
                "measurement": "phase_aligned_cross_sensor_observation",
                "rationale": "Competing hypotheses are close; phase-aligned sensing improves discrimination.",
                "expected_uncertainty_reduction": round(_clip01(0.45 + 0.35 * observability_gap), 4),
            }
        )

    return {
        "recommended_measurements": recommendations,
        "observability_gap_score": round(observability_gap, 4),
        "expected_uncertainty_reduction": round(max(r["expected_uncertainty_reduction"] for r in recommendations), 4),
        "advisory": True,
        "status": "experimental",
    }
