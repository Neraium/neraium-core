from __future__ import annotations

from typing import Any

from .branching import StructuralBranchingEngine
from .replay import StructuralReplayEngine
from .simulation import StructuralTransitionSimulator
from .what_if import StructuralWhatIfEngine
from .world_models import StructuralWorldModelEngine


class StructuralSandboxEngine:
    """Experimental structural sandbox for replaying and comparing futures."""

    def __init__(self) -> None:
        self.simulator = StructuralTransitionSimulator()
        self.branching = StructuralBranchingEngine(self.simulator)
        self.replay = StructuralReplayEngine(self.simulator)
        self.world_models = StructuralWorldModelEngine(self.simulator)
        self.what_if = StructuralWhatIfEngine()

    def evaluate(
        self,
        *,
        latent_state: dict[str, Any],
        transition: dict[str, Any],
        trajectory: dict[str, Any],
        laws: dict[str, Any],
        intervention: dict[str, Any],
        reliability: dict[str, Any],
    ) -> dict[str, Any]:
        embedding = list(latent_state.get("embedding") or [])
        observed_trajectory = list(latent_state.get("trajectory") or [])
        if not embedding and observed_trajectory:
            embedding = list(observed_trajectory[-1])
        if not embedding:
            return {
                "structural_sandbox": {
                    "status": "experimental",
                    "available": False,
                    "reason": "missing_latent_state",
                    "can_drive_operator_recommendations": False,
                }
            }

        sim_reliability = self._simulation_reliability(trajectory=trajectory, laws=laws, reliability=reliability)
        current = self.simulator.simulate(
            initial_embedding=embedding,
            transition=transition,
            trajectory=trajectory,
            law_candidate=((laws.get("law_candidates") or [{}])[:1] or [{}])[0],
            reliability=sim_reliability,
            mode="baseline",
        )
        branches = self.branching.build_branches(
            initial_embedding=embedding,
            transition=transition,
            trajectory=trajectory,
            laws=laws,
            intervention=intervention,
            reliability=sim_reliability,
        )
        replay = self.replay.replay(
            observed_trajectory=observed_trajectory,
            transition=transition,
            trajectory=trajectory,
            laws=laws,
            intervention=intervention,
            reliability=sim_reliability,
        )
        observed_future = observed_trajectory[-4:] if len(observed_trajectory) >= 4 else None
        world_models = self.world_models.compare(
            initial_embedding=embedding,
            transition=transition,
            trajectory=trajectory,
            laws=laws,
            reliability=sim_reliability,
            observed_future=observed_future,
        )
        baseline_branch = (branches.get("scenario_branches") or [{}])[0]
        what_if = self.what_if.generate(
            baseline_branch=baseline_branch,
            trajectory=trajectory,
            transition=transition,
        )

        return {
            "structural_sandbox": {
                "current_state_simulation": current,
                "scenario_branches": branches.get("scenario_branches", []),
                "branch_comparison": branches.get("branch_comparison", {}),
                "dominant_divergence_factors": branches.get("dominant_divergence_factors", []),
                "replay_trace": replay.get("replay_trace", []),
                "prediction_error_trace": replay.get("prediction_error_trace", []),
                "law_support_trace": replay.get("law_support_trace", []),
                "intervention_effect_trace": replay.get("intervention_effect_trace", []),
                "world_models": world_models.get("competing_world_models", []),
                "world_model_fit_scores": world_models.get("world_model_fit_scores", {}),
                "best_explaining_model": world_models.get("best_explaining_model"),
                "discriminating_observations": world_models.get("discriminating_observations", []),
                "divergence_analysis": replay.get("divergence_analysis", {}),
                "what_if_analysis": what_if.get("what_if_scenarios", []),
                "scenario_delta_vs_baseline": what_if.get("scenario_delta_vs_baseline", {}),
                "confidence_and_assumptions": what_if.get("confidence_and_assumptions", {}),
                "simulation_reliability": sim_reliability,
                "status": "experimental",
                "can_drive_operator_recommendations": False,
            }
        }

    def _simulation_reliability(
        self,
        *,
        trajectory: dict[str, Any],
        laws: dict[str, Any],
        reliability: dict[str, Any],
    ) -> dict[str, Any]:
        trajectory_novelty = float(trajectory.get("novelty_score", 0.5))
        branch_support = float(trajectory.get("family_similarity", 0.5))
        law_support = float((((laws.get("law_candidates") or [{}])[:1] or [{}])[0]).get("confidence", 0.0))
        base_conf = float((((reliability.get("transition_dynamics") or {}).get("transition_confidence", 0.6))) if isinstance(reliability, dict) else 0.6)
        novelty_penalty = 0.45 * trajectory_novelty
        transfer_penalty = max(0.0, 0.3 - branch_support * 0.3)
        conf = max(0.05, min(0.95, base_conf - novelty_penalty - transfer_penalty + law_support * 0.25))
        return {
            "simulation_confidence": round(conf, 6),
            "branch_support": round(branch_support, 6),
            "law_support": round(law_support, 6),
            "novelty_penalties": round(novelty_penalty, 6),
            "transfer_penalties": round(transfer_penalty, 6),
            "degraded": conf < 0.4,
        }
