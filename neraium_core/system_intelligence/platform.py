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
        self._real_world_validation: dict[str, Any] = {
            "decision_accuracy": 0.0,
            "harm_rate": 0.0,
            "calibration": 0.0,
            "law_validation": {},
            "drift_signals": {},
        }

    def set_real_world_validation(self, payload: dict[str, Any]) -> None:
        self._real_world_validation = {
            "decision_accuracy": float(payload.get("decision_accuracy", 0.0)),
            "harm_rate": float(payload.get("harm_rate", 0.0)),
            "calibration": float(payload.get("calibration", 0.0)),
            "law_validation": dict(payload.get("law_validation") or {}),
            "drift_signals": dict(payload.get("drift_signals") or {}),
        }

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
        # Production boundary hardening:
        # Operator-facing compatibility output must remain driven by production-grade layers only.
        compatibility = to_operator_compatibility(production)

        return {
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
            "real_world_validation": self._real_world_validation,
            **merged,
        }
