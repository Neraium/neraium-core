from __future__ import annotations

from typing import Any

import numpy as np


class TrajectoryConditionedForecaster:
    """Forecasts likely path evolution from trajectory-family evidence with explicit uncertainty."""

    def forecast(
        self,
        *,
        trajectory_intelligence: dict[str, Any],
        transition_dynamics: dict[str, Any],
    ) -> dict[str, Any]:
        if trajectory_intelligence.get("status") != "ready":
            return {
                "status": "warming",
                "likely_next_path_family": "unknown",
                "trajectory_conditioned_escalation_probability": transition_dynamics.get("escalation_probability", 0.0),
                "estimated_steps_to_critical_region": {"median": None, "range": [None, None]},
                "matched_historical_progressions": [],
                "uncertainty": 0.85,
                "notes": "Trajectory-conditioned forecast unavailable until trajectory memory warms up.",
            }

        nearest = list(trajectory_intelligence.get("nearest_trajectory_archetypes") or [])
        if not nearest:
            return {
                "status": "warming",
                "likely_next_path_family": transition_dynamics.get("transition_path", "unknown"),
                "trajectory_conditioned_escalation_probability": transition_dynamics.get("escalation_probability", 0.0),
                "estimated_steps_to_critical_region": {"median": None, "range": [None, None]},
                "matched_historical_progressions": [],
                "uncertainty": 0.8,
                "notes": "No matched trajectory archetypes yet.",
            }

        sims = np.asarray([float(item.get("similarity", 0.0)) for item in nearest], dtype=float)
        if float(np.sum(sims)) <= 1e-9:
            sims = np.ones_like(sims)
        weights = sims / float(np.sum(sims))
        escalation_freq = np.asarray([float(item.get("historical_escalation_frequency", 0.0)) for item in nearest], dtype=float)
        traj_escalation = float(np.sum(weights * escalation_freq))
        base_escalation = float(transition_dynamics.get("escalation_probability", 0.0))

        support = np.asarray([max(1.0, float(item.get("support", 1))) for item in nearest], dtype=float)
        support_score = float(np.sum(weights * np.minimum(1.0, support / 15.0)))
        blend_w = 0.30 + 0.55 * support_score
        conditioned_prob = blend_w * traj_escalation + (1.0 - blend_w) * base_escalation

        progressions = list(trajectory_intelligence.get("matched_historical_progressions") or [])
        likely_next = "unknown"
        if progressions:
            likely_next = str(max(progressions, key=lambda item: float(item.get("frequency", 0.0))).get("path_family", "unknown"))
        elif float(conditioned_prob) > 0.65:
            likely_next = "escalating"
        else:
            likely_next = str(trajectory_intelligence.get("current_trajectory_path_family", "drifting"))

        # interpretable proxy: fewer steps to critical when trajectory-conditioned risk is high.
        median_steps = int(round(2.0 + 10.0 * (1.0 - conditioned_prob)))
        spread = int(round(2.0 + 4.0 * (1.0 - support_score)))
        lo = max(1, median_steps - spread)
        hi = median_steps + spread
        uncertainty = float(max(0.1, min(0.9, 1.0 - (0.65 * support_score + 0.35 * float(np.max(weights))))))

        return {
            "status": "ready",
            "likely_next_path_family": likely_next,
            "trajectory_conditioned_escalation_probability": round(float(conditioned_prob), 6),
            "estimated_steps_to_critical_region": {"median": int(median_steps), "range": [int(lo), int(hi)]},
            "matched_historical_progressions": progressions,
            "uncertainty": round(uncertainty, 6),
            "notes": "Trajectory forecast is evidence-weighted from nearest archetype histories; uncertainty increases with weak support.",
        }
