from __future__ import annotations

from typing import Any

import numpy as np


class TrajectoryConditionedForecaster:
    """Forecast likely path evolution from trajectory-family evidence with explicit uncertainty."""

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
                "uncertainty": 0.9,
                "novelty_aware_confidence_penalty": 0.2,
                "notes": "Trajectory-conditioned forecast unavailable until trajectory memory warms up.",
            }

        nearest = list(
            trajectory_intelligence.get("nearest_trajectory_families")
            or trajectory_intelligence.get("nearest_trajectory_archetypes")
            or []
        )
        if not nearest:
            return {
                "status": "warming",
                "likely_next_path_family": transition_dynamics.get("transition_path", "unknown"),
                "trajectory_conditioned_escalation_probability": transition_dynamics.get("escalation_probability", 0.0),
                "estimated_steps_to_critical_region": {"median": None, "range": [None, None]},
                "matched_historical_progressions": [],
                "uncertainty": 0.86,
                "novelty_aware_confidence_penalty": 0.18,
                "notes": "No matched trajectory families yet.",
            }

        sims = np.asarray(
            [float(item.get("family_similarity", item.get("similarity", 0.0))) for item in nearest],
            dtype=float,
        )
        if float(np.sum(sims)) <= 1e-9:
            sims = np.ones_like(sims)
        weights = sims / float(np.sum(sims))

        escalation_freq = np.asarray([float(item.get("escalation_frequency", item.get("historical_escalation_frequency", 0.0))) for item in nearest], dtype=float)
        traj_escalation = float(np.sum(weights * escalation_freq))
        base_escalation = float(transition_dynamics.get("escalation_probability", 0.0))

        supports = np.asarray([max(1.0, float(item.get("support", 1))) for item in nearest], dtype=float)
        support_score = float(np.sum(weights * np.minimum(1.0, supports / 18.0)))
        novelty = float(trajectory_intelligence.get("novelty_score", 0.0))
        novelty_penalty = 0.45 * novelty

        blend_w = 0.25 + 0.60 * support_score
        raw_prob = blend_w * traj_escalation + (1.0 - blend_w) * base_escalation
        conditioned_prob = max(0.0, min(1.0, raw_prob * (1.0 - 0.35 * novelty_penalty) + base_escalation * 0.35 * novelty_penalty))

        progressions = list(trajectory_intelligence.get("matched_historical_progressions") or [])
        likely_next = transition_dynamics.get("transition_path", "unknown")
        if progressions:
            likely_next = str(max(progressions, key=lambda item: float(item.get("frequency", 0.0))).get("path_family", likely_next))

        median_steps = int(round(2.0 + 11.0 * (1.0 - conditioned_prob)))
        spread = int(round(2.0 + 5.5 * (1.0 - support_score + novelty_penalty)))
        lo = max(1, median_steps - spread)
        hi = median_steps + spread

        uncertainty = float(max(0.08, min(0.96, 1.0 - (0.62 * support_score + 0.25 * float(np.max(weights))) + novelty_penalty)))
        return {
            "status": "ready",
            "likely_next_path_family": likely_next,
            "trajectory_conditioned_escalation_probability": round(float(conditioned_prob), 6),
            "estimated_steps_to_critical_region": {"median": int(median_steps), "range": [int(lo), int(hi)]},
            "matched_historical_progressions": progressions,
            "uncertainty": round(uncertainty, 6),
            "novelty_aware_confidence_penalty": round(float(novelty_penalty), 6),
            "notes": "Trajectory-family-conditioned forecast with support weighting and novelty-aware confidence penalty.",
        }
