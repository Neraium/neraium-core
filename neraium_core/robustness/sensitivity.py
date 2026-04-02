from __future__ import annotations

from typing import Dict, List

import numpy as np


def _score(matrix: np.ndarray) -> float:
    safe = np.nan_to_num(np.asarray(matrix, dtype=float), nan=0.0)
    if safe.ndim != 2 or safe.shape[0] < 2:
        return 0.0
    drift = np.linalg.norm(safe[-1] - safe[0])
    vol = float(np.mean(np.std(safe, axis=0)))
    return float(drift + vol)


def compute_sensitivity(*, baseline_window: np.ndarray, recent_window: np.ndarray, feature_names: List[str], trajectory_score: float, lock_in_score: float, branching_score: float) -> Dict[str, object]:
    base_score = _score(recent_window) + 1e-9
    contributions: Dict[str, float] = {}

    safe_recent = np.nan_to_num(np.asarray(recent_window, dtype=float), nan=0.0)
    safe_baseline = np.nan_to_num(np.asarray(baseline_window, dtype=float), nan=0.0)

    for idx, name in enumerate(feature_names):
        if idx >= safe_recent.shape[1]:
            break
        lofo = np.array(safe_recent, copy=True)
        lofo[:, idx] = np.mean(lofo[:, idx])
        delta = max(0.0, base_score - _score(lofo))
        contributions[name] = float(delta / base_score)

    sorted_items = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    top = [{"feature": k, "contribution": round(v, 4)} for k, v in sorted_items[:5]]

    traj = {k: round(v * trajectory_score, 4) for k, v in sorted_items[:8]}
    lockin = {k: round(v * lock_in_score, 4) for k, v in sorted_items[:8]}
    branch = {k: round(v * branching_score, 4) for k, v in sorted_items[:8]}

    directional = np.mean(safe_recent[-1] - safe_baseline[-1][: safe_recent.shape[1]]) if safe_baseline.ndim == 2 and safe_baseline.shape[1] >= safe_recent.shape[1] else 0.0
    residual = float(np.mean(np.abs(safe_recent - np.mean(safe_recent, axis=0, keepdims=True))))
    local_impact = float(np.mean([v for _k, v in sorted_items[:3]]) if sorted_items else 0.0)

    grouped = {
        "leave_one_group_out": {
            "group_a": round(float(np.mean(list(contributions.values())[: max(1, len(contributions)//2)] or [0.0])), 4),
            "group_b": round(float(np.mean(list(contributions.values())[max(1, len(contributions)//2):] or [0.0])), 4),
        },
        "directional_drift_contribution": round(float(abs(directional)), 4),
        "residual_contribution": round(residual / (base_score + 1e-9), 4),
        "local_perturbation_impact": round(local_impact, 4),
    }

    return {
        "top_drivers": top,
        "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
        "trajectory_drivers": traj,
        "lock_in_drivers": lockin,
        "branching_drivers": branch,
        "horizon_drivers": {k: round(v * 0.5, 4) for k, v in sorted_items[:8]},
        "counterfactual_drivers": {k: round(v * 0.5, 4) for k, v in sorted_items[:8]},
        "group_contributions": grouped,
    }
