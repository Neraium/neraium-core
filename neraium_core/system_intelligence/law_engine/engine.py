from __future__ import annotations

from typing import Any

from neraium_core.system_intelligence.law_decision.decision_support import apply_law_decision_support
from neraium_core.system_intelligence.law_gating.gating import evaluate_law_candidates
from neraium_core.system_intelligence.law_governance import StructuralLawGovernance


class StructuralLawDecisionEngine:
    """Bounded law layer that turns candidate laws into transparent decision-support signals."""

    def __init__(self) -> None:
        self.governance = StructuralLawGovernance()

    def evaluate(
        self,
        *,
        law_candidates: dict[str, Any],
        trajectory_info: dict[str, Any],
        mechanism_info: dict[str, Any],
        risk_assessment: dict[str, Any],
        operator_guidance: dict[str, Any],
        counterfactuals: dict[str, Any],
        reliability: dict[str, Any] | None = None,
        falsification: dict[str, Any] | None = None,
        intervention_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidates = list(law_candidates.get("law_candidates") or [])
        governance = self.governance.evaluate(
            law_candidates=candidates,
            trajectory_info=trajectory_info,
            reliability=reliability,
            falsification=falsification,
            intervention_info=intervention_info or counterfactuals,
            allow_auto_promotion=True,
        )
        gating = evaluate_law_candidates(
            law_candidates=candidates,
            trajectory_info=trajectory_info,
            mechanism_info=mechanism_info,
            governance=governance,
        )
        decision = apply_law_decision_support(
            gating=gating,
            risk_assessment=risk_assessment,
            operator_guidance=operator_guidance,
            counterfactuals=counterfactuals,
            trajectory_info=trajectory_info,
        )
        return {
            "status": "ready" if law_candidates.get("status") == "ready" else str(law_candidates.get("status", "warming")),
            "structural_law_governance": governance,
            **gating,
            **decision,
        }
