from __future__ import annotations

import math
from typing import Any


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _response_for_hypothesis(hypothesis: str, experiment: dict[str, Any]) -> float:
    name = str(experiment.get("name", ""))
    base = 0.5
    if "Drift" in hypothesis and "damp" in name:
        base = 0.25
    elif "Oscillatory" in hypothesis and "damp" in name:
        base = 0.75
    elif "Localized" in hypothesis and "sensor" in name:
        base = 0.8
    elif "Drift" in hypothesis and "sensor" in name:
        base = 0.45
    elif "Oscillatory" in hypothesis and "phase" in name:
        base = 0.8
    elif "Drift" in hypothesis and "phase" in name:
        base = 0.35
    return base


def identify_discriminative_experiments(
    *,
    hypotheses: list[dict[str, Any]],
    uncertainty: dict[str, Any],
    intervention_ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    candidates = [
        {
            "name": "temporarily_damp_subsystem_X",
            "type": "intervention",
            "safety": "low_risk_reversible",
        },
        {
            "name": "increase_sampling_rate_sensor_Y",
            "type": "measurement",
            "safety": "low_risk_observational",
        },
        {
            "name": "phase_response_probe_relationship_pair",
            "type": "measurement",
            "safety": "low_risk_observational",
        },
    ]
    for rank in intervention_ranking[:2]:
        nm = str(rank.get("name") or rank.get("intervention_type") or "candidate_intervention")
        candidates.append({"name": nm, "type": "intervention", "safety": "bounded_existing"})

    for cand in candidates:
        responses = [_response_for_hypothesis(str(h.get("hypothesis", "")), cand) for h in hypotheses]
        if not responses:
            continue
        mean = sum(responses) / len(responses)
        variance = sum((r - mean) ** 2 for r in responses) / len(responses)
        separation = _clip01(math.sqrt(variance) * 2.0)
        magnitude = float((uncertainty.get("uncertainty_magnitude") or 0.5))
        info_gain = _clip01(0.65 * separation + 0.35 * magnitude)
        experiments.append(
            {
                "name": cand["name"],
                "experiment_type": cand["type"],
                "safety_profile": cand["safety"],
                "expected_information_gain": round(info_gain, 4),
                "hypothesis_separation_score": round(separation, 4),
                "discriminative_power": round(_clip01(0.7 * separation + 0.3 * info_gain), 4),
                "expected_outcomes": {
                    str(h["hypothesis"]): round(resp, 4) for h, resp in zip(hypotheses, responses)
                },
            }
        )

    experiments.sort(key=lambda e: (e["expected_information_gain"], e["hypothesis_separation_score"]), reverse=True)
    return {
        "candidate_experiments": experiments,
        "expected_information_gain": experiments[0]["expected_information_gain"] if experiments else 0.0,
        "discriminative_power": experiments[0]["discriminative_power"] if experiments else 0.0,
    }
