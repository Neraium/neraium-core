"""Engine configuration for agnostic runner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """Configuration for Neraium engine components.

    Attributes:
        mode: Operating mode ('production', 'research_assistive', 'experimental', or 'full')
        enable_trajectory_memory: Track system trajectory history
        enable_law_extraction: Extract structural laws
        enable_law_engine: Run law engine for recommendations
        enable_intervention_memory: Track intervention history
        enable_reliability_calibration: Calibrate confidence scores
        enable_cross_system_transfer: Use cross-system intelligence
        enable_advisory_layers: Run advisory recommendation layers
        enable_experimental_layers: Run experimental layers
    """

    mode: str = "production"
    enable_trajectory_memory: bool = True
    enable_law_extraction: bool = True
    enable_law_engine: bool = True
    enable_intervention_memory: bool = True
    enable_reliability_calibration: bool = True
    enable_cross_system_transfer: bool = True
    enable_advisory_layers: bool = True
    enable_experimental_layers: bool = False

    def to_component_flags(self) -> dict[str, bool]:
        """Convert to component flags dict for NeraiumModel."""
        return {
            "trajectory_intelligence": self.enable_trajectory_memory,
            "law_engine": self.enable_law_extraction and self.enable_law_engine,
            "intervention_memory": self.enable_intervention_memory,
            "reliability_calibration": self.enable_reliability_calibration,
            "cross_system_transfer": self.enable_cross_system_transfer,
            "advisory_layers": self.enable_advisory_layers,
            "experimental_layers": self.enable_experimental_layers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EngineConfig:
        """Create config from dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def production(cls) -> EngineConfig:
        """Preset for production mode."""
        return cls(
            mode="production",
            enable_trajectory_memory=True,
            enable_law_extraction=False,
            enable_law_engine=False,
            enable_intervention_memory=True,
            enable_reliability_calibration=True,
            enable_cross_system_transfer=False,
            enable_advisory_layers=True,
            enable_experimental_layers=False,
        )

    @classmethod
    def research(cls) -> EngineConfig:
        """Preset for research mode (all components enabled)."""
        return cls(
            mode="research_assistive",
            enable_trajectory_memory=True,
            enable_law_extraction=True,
            enable_law_engine=True,
            enable_intervention_memory=True,
            enable_reliability_calibration=True,
            enable_cross_system_transfer=True,
            enable_advisory_layers=True,
            enable_experimental_layers=False,
        )

    @classmethod
    def experimental(cls) -> EngineConfig:
        """Preset for experimental mode (all components)."""
        return cls(
            mode="full",
            enable_trajectory_memory=True,
            enable_law_extraction=True,
            enable_law_engine=True,
            enable_intervention_memory=True,
            enable_reliability_calibration=True,
            enable_cross_system_transfer=True,
            enable_advisory_layers=True,
            enable_experimental_layers=True,
        )

    @classmethod
    def minimal(cls) -> EngineConfig:
        """Preset for minimal mode (core only)."""
        return cls(
            mode="production",
            enable_trajectory_memory=False,
            enable_law_extraction=False,
            enable_law_engine=False,
            enable_intervention_memory=False,
            enable_reliability_calibration=False,
            enable_cross_system_transfer=False,
            enable_advisory_layers=False,
            enable_experimental_layers=False,
        )
