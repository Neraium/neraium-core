from __future__ import annotations

from typing import Any, Literal

from .adapters.compatibility import to_operator_compatibility
from .capability_boundaries import boundaries_for_sections
from .orchestration import ProductionIntelligenceOrchestrator, ResearchExperimentalOrchestrator

OperatingMode = Literal["production", "research_assistive", "experimental", "full"]


class StructuralSystemIntelligencePlatform:
    """Layered structural intelligence with explicit production/advisory/experimental boundaries."""

    def __init__(self, *, operating_mode: OperatingMode = "production") -> None:
        self.operating_mode: OperatingMode = operating_mode
        self.production = ProductionIntelligenceOrchestrator()
        self.research = ResearchExperimentalOrchestrator()

    def update(self, observation: dict[str, Any], *, operating_mode: OperatingMode | None = None) -> dict[str, Any]:
        mode = operating_mode or self.operating_mode
        include_advisory = mode in {"research_assistive", "experimental", "full"}
        include_experimental = mode in {"experimental", "full"}

        production = self.production.update(observation)
        advisory: dict[str, Any] = {}
        experimental: dict[str, Any] = {}

        if include_advisory or include_experimental:
            advisory, experimental = self.research.update(
                observation=observation,
                production=production,
                include_experimental=include_experimental,
            )

        merged: dict[str, Any] = {**production, **advisory, **experimental}
        compatibility = to_operator_compatibility(merged)

        out = {
            "operating_mode": mode,
            "production_intelligence": production,
            "advisory_intelligence": advisory,
            "experimental_intelligence": experimental,
            "capability_boundaries": boundaries_for_sections(
                sections={
                    "production": {**production, "compatibility": compatibility},
                    "advisory": advisory,
                    "experimental": experimental,
                }
            ),
            "compatibility": compatibility,
            "compatibility_aliases": {
                "legacy_top_level_sections": sorted(merged.keys()),
                "note": "Top-level legacy sections are preserved for transition compatibility.",
            },
            **merged,
        }
        return out
