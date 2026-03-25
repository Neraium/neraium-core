from __future__ import annotations

from typing import Dict


def generate_structural_explanations(*, trajectory_label: str, branching_count: float, lock_in_score: float, horizon_label: str, counterfactual_spread: float, drift_type: str, stability: Dict[str, float], path_prototype: str = "unknown") -> Dict[str, str]:
    return {
        "trajectory": f"trajectory={trajectory_label}; trajectory_stability={stability.get('trajectory_stability_score', 0.0):.2f}",
        "branching": f"branch_count={branching_count:.2f}; branching_stability={stability.get('branching_stability_score', 0.0):.2f}",
        "constraint": f"lock_in={lock_in_score:.2f}; lock_in_stability={stability.get('lock_in_stability_score', 0.0):.2f}",
        "horizon": f"horizon={horizon_label}; horizon_stability={stability.get('horizon_stability_score', 0.0):.2f}",
        "drift_noise": f"interpreted_change_type={drift_type}; noise_to_drift={stability.get('window_to_window_consistency', 0.0):.2f}",
        "path_prototype": f"dominant_prototype={path_prototype}; counterfactual_spread={counterfactual_spread:.2f}",
    }
