from __future__ import annotations

from typing import Any


def build_experiment_plan(
    *,
    hypotheses: list[dict[str, Any]],
    candidate_experiments: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    uncertainty: dict[str, Any],
) -> dict[str, Any]:
    top = candidate_experiments[0] if candidate_experiments else {}
    h0 = hypotheses[0] if hypotheses else {"hypothesis": "hypothesis_A"}
    h1 = hypotheses[1] if len(hypotheses) > 1 else {"hypothesis": "hypothesis_B"}
    alt = measurements[0] if measurements else {"measurement": "increase_sampling_frequency::subsystem_x"}

    expected_outcomes = top.get("expected_outcomes") or {
        str(h0.get("hypothesis", "hypothesis_A")): "moderate response",
        str(h1.get("hypothesis", "hypothesis_B")): "strong response",
    }
    risk_level = "low" if "low_risk" in str(top.get("safety_profile", "")) else "guarded"

    return {
        "experiment_plan": {
            "objective": f"disambiguate {h0.get('hypothesis', 'hypothesis_A')} vs {h1.get('hypothesis', 'hypothesis_B')}",
            "recommended_action": top.get("name", "temporarily_damp_subsystem_X"),
            "alternative_measurement": alt.get("measurement", "increase_sampling_rate_sensor_Y"),
            "expected_outcomes": expected_outcomes,
            "expected_information_gain": round(float(top.get("expected_information_gain", 0.0)), 4),
            "uncertainty_magnitude": round(float(uncertainty.get("uncertainty_magnitude", 0.0)), 4),
            "risk_level": risk_level,
            "advisory": True,
            "status": "experimental",
        }
    }
