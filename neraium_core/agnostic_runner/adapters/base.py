"""Base classes for data source and output sink adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass
class Frame:
    """A single frame of sensor data to process."""
    timestamp: float
    unit_id: str
    sensors: dict[str, float]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert frame to dictionary for engine processing."""
        return {
            "timestamp": self.timestamp,
            "unit_id": self.unit_id,
            "sensors": self.sensors,
            **(self.metadata or {}),
        }


@dataclass
class Result:
    """Result from processing a single frame."""
    frame: Frame
    output: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass
class RunContext:
    """Context information for a run."""
    run_id: str
    run_mode: str  # 'production', 'research_assistive', 'experimental'
    start_time: float
    data_source_type: str
    output_sink_type: str
    frame_count: int = 0
    error_count: int = 0


class DataSourceAdapter(ABC):
    """Base class for data source adapters."""

    def __init__(self, source_config: dict[str, Any] | None = None) -> None:
        """Initialize adapter with configuration."""
        self.source_config = source_config or {}

    @abstractmethod
    def read_frames(self) -> Iterator[Frame]:
        """Yield frames from data source.

        Raises:
            RuntimeError: If source cannot be read or is invalid.
        """
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Validate that the data source is accessible and valid."""
        ...

    def close(self) -> None:
        """Clean up resources (override if needed)."""
        pass


class OutputSink(ABC):
    """Base class for output sink adapters."""

    def __init__(self, sink_config: dict[str, Any] | None = None) -> None:
        """Initialize adapter with configuration."""
        self.sink_config = sink_config or {}
        self.result_count = 0

    @abstractmethod
    def write_result(self, result: Result) -> None:
        """Write a single result to output destination.

        Raises:
            RuntimeError: If write fails.
        """
        ...

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Finalize output and return summary metadata.

        Called once after all results are written.
        Should return dict with keys like 'output_path', 'record_count', etc.
        """
        ...

    def validate(self) -> bool:
        """Validate that the sink is writable and configured correctly."""
        return True

    def close(self) -> None:
        """Clean up resources (override if needed)."""
        pass
