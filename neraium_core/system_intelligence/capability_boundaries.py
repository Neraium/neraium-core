from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapabilityStatus = Literal["production", "advisory", "experimental"]
DependencyLevel = Literal["core", "supporting", "optional"]
MaturityLevel = Literal["production", "pilot", "research"]
ActionabilityLevel = Literal["actionable", "advisory_only", "inspect_only"]


@dataclass(frozen=True)
class CapabilityBoundary:
    capability_key: str
    status: CapabilityStatus
    can_drive_operator_recommendations: bool
    dependency_level: DependencyLevel
    maturity_level: MaturityLevel
    actionability_level: ActionabilityLevel

    def to_dict(self) -> dict[str, object]:
        return {
            "capability_key": self.capability_key,
            "status": self.status,
            "can_drive_operator_recommendations": self.can_drive_operator_recommendations,
            "dependency_level": self.dependency_level,
            "maturity_level": self.maturity_level,
            "actionability_level": self.actionability_level,
        }


CAPABILITY_BOUNDARIES: dict[str, CapabilityBoundary] = {
    "latent_structural_state": CapabilityBoundary("latent_structural_state", "production", True, "core", "production", "actionable"),
    "transition_dynamics": CapabilityBoundary("transition_dynamics", "production", True, "core", "production", "actionable"),
    "counterfactuals": CapabilityBoundary("counterfactuals", "production", True, "supporting", "pilot", "actionable"),
    "trajectory_archetypes": CapabilityBoundary("trajectory_archetypes", "production", True, "core", "production", "actionable"),
    "trajectory_forecast": CapabilityBoundary("trajectory_forecast", "production", True, "supporting", "production", "actionable"),
    "intervention_intelligence": CapabilityBoundary("intervention_intelligence", "production", True, "core", "production", "actionable"),
    "reliability_intelligence": CapabilityBoundary("reliability_intelligence", "production", True, "core", "production", "actionable"),
    "mechanism_discovery": CapabilityBoundary("mechanism_discovery", "advisory", False, "supporting", "pilot", "advisory_only"),
    "structural_law_candidates": CapabilityBoundary("structural_law_candidates", "advisory", False, "optional", "pilot", "advisory_only"),
    "law_engine_decision": CapabilityBoundary("law_engine_decision", "advisory", False, "optional", "pilot", "advisory_only"),
    "cross_system_structural_intelligence": CapabilityBoundary("cross_system_structural_intelligence", "advisory", False, "optional", "pilot", "advisory_only"),
    "experimental_universal_structural_intelligence": CapabilityBoundary("experimental_universal_structural_intelligence", "experimental", False, "optional", "research", "inspect_only"),
    "falsification_analysis": CapabilityBoundary("falsification_analysis", "experimental", False, "optional", "research", "inspect_only"),
    "active_learning": CapabilityBoundary("active_learning", "experimental", False, "optional", "research", "inspect_only"),
    "structural_sandbox": CapabilityBoundary("structural_sandbox", "experimental", False, "optional", "research", "inspect_only"),
    "compatibility": CapabilityBoundary("compatibility", "production", True, "core", "production", "actionable"),
}


def boundaries_for_sections(*, sections: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    active: dict[str, dict[str, object]] = {}
    for payload in sections.values():
        for key in payload.keys():
            boundary = CAPABILITY_BOUNDARIES.get(key)
            if boundary is not None:
                active[key] = boundary.to_dict()
    return active
