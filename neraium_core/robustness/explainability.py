from __future__ import annotations

from typing import Dict


def generate_structural_explanations(*, trajectory_label: str, branching_count: float, lock_in_score: float, horizon_label: str, counterfactual_spread: float, drift_type: str, stability: Dict[str, float]) -> Dict[str, str]:
    return {
        "trajectory": (
            f"Trajectory is {trajectory_label} because run/window consistency is {stability.get('trajectory_stability_score', 0.0):.2f}."
        ),
        "branching": (
            f"Branching evidence is {'present' if branching_count > 1.25 else 'limited'} with estimated branch count {branching_count:.2f}."
        ),
        "constraint": (
            f"Lock-in intensity is {lock_in_score:.2f} and drift/noise interpretation is {drift_type}."
        ),
        "horizon": f"Horizon classification is {horizon_label} with volatility {stability.get('horizon_volatility', 0.0):.2f}.",
        "counterfactual": (
            f"Counterfactual divergence spread is {counterfactual_spread:.2f}; higher spread indicates structurally distinct futures."
        ),
    }
