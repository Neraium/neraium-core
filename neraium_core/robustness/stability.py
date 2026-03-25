from __future__ import annotations

from typing import Dict

import numpy as np


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def compute_stability_metrics(*, drift_history: list[float], transition_history: list[float], state_history: list[str], trajectory_label: str, branch_count: float, lock_in_score: float, horizon_steps: float, counterfactual_spread: float) -> Dict[str, float]:
    d = np.asarray(drift_history[-20:], dtype=float)
    t = np.asarray(transition_history[-20:], dtype=float)

    run_to_run_consistency = 1.0 - min(1.0, float(np.std(d))) if d.size else 1.0
    window_to_window_consistency = 1.0 - min(1.0, float(np.mean(np.abs(np.diff(d))))) if d.size >= 2 else 1.0
    trajectory_stability_score = _clamp01(0.5 * run_to_run_consistency + 0.5 * window_to_window_consistency)

    branching_consistency_score = _clamp01(1.0 - (min(3.0, max(0.0, branch_count - 1.0)) / 3.0))
    lock_in_variance = float(np.var(t)) if t.size > 2 else float(lock_in_score * 0.05)
    horizon_volatility = float(np.std(d[-6:])) if d.size >= 6 else float(np.std(d)) if d.size else 0.0
    counterfactual_consistency = _clamp01(1.0 - min(1.0, float(counterfactual_spread)))

    if state_history:
        dominant = max(set(state_history), key=state_history.count)
        state_consistency = float(state_history.count(dominant)) / float(len(state_history))
    else:
        state_consistency = 1.0

    overall = _clamp01(
        0.2 * trajectory_stability_score
        + 0.15 * branching_consistency_score
        + 0.15 * counterfactual_consistency
        + 0.15 * state_consistency
        + 0.2 * (1.0 - min(1.0, lock_in_variance))
        + 0.15 * (1.0 - min(1.0, horizon_volatility))
    )

    return {
        "trajectory_stability_score": round(trajectory_stability_score, 4),
        "branching_consistency_score": round(branching_consistency_score, 4),
        "lock_in_variance": round(float(lock_in_variance), 4),
        "horizon_volatility": round(float(horizon_volatility), 4),
        "counterfactual_consistency": round(counterfactual_consistency, 4),
        "run_to_run_consistency": round(run_to_run_consistency, 4),
        "window_to_window_consistency": round(window_to_window_consistency, 4),
        "overall_stability": round(overall, 4),
        "trajectory_label": str(trajectory_label),
    }
