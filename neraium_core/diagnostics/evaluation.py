from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import numpy as np


@dataclass
class EvalSummary:
    detection_timing: float
    structural_divergence: float
    branch_diversity: float
    lock_in_onset_timing: float
    horizon_transition_timing: float
    advanced_population_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "detection_timing": round(self.detection_timing, 4),
            "structural_divergence": round(self.structural_divergence, 4),
            "branch_diversity": round(self.branch_diversity, 4),
            "lock_in_onset_timing": round(self.lock_in_onset_timing, 4),
            "horizon_transition_timing": round(self.horizon_transition_timing, 4),
            "advanced_population_rate": round(self.advanced_population_rate, 4),
        }


def summarize_results(results: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(results)
    if not rows:
        return EvalSummary(0, 0, 0, 0, 0, 0).to_dict()

    drift = np.asarray([float(r.get("structural_drift_score", 0.0) or 0.0) for r in rows], dtype=float)
    states = [str(r.get("state", "STABLE")) for r in rows]
    first_watch = next((i for i, s in enumerate(states) if s in {"WATCH", "ALERT"}), len(rows))

    branches = []
    lockins = []
    horizons = []
    advanced_pop = []
    for r in rows:
        exp = r.get("experimental_analytics", {}) or {}
        b = (exp.get("branching_analysis", {}) or {}).get("branch_count_estimate")
        lock_in_score = (exp.get("constraint_analysis", {}) or {}).get("lock_in_score")
        h = (exp.get("horizon_analysis", {}) or {}).get("horizon_steps")
        branches.append(float(b or 1.0))
        lockins.append(float(lock_in_score or 0.0))
        horizons.append(float(h or 0.0))
        advanced_pop.append(1.0 if isinstance(exp, dict) and len(exp) > 0 else 0.0)

    lock_idx = next((i for i, v in enumerate(lockins) if v > 0.6), len(rows))
    hz_idx = next((i for i, v in enumerate(horizons) if v > 0.0), len(rows))

    return EvalSummary(
        detection_timing=float(first_watch),
        structural_divergence=float(np.mean(np.abs(np.diff(drift)))) if drift.size > 1 else 0.0,
        branch_diversity=float(np.std(branches)),
        lock_in_onset_timing=float(lock_idx),
        horizon_transition_timing=float(hz_idx),
        advanced_population_rate=float(np.mean(advanced_pop)),
    ).to_dict()
