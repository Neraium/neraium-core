"""Agnostic runner for Neraium: pluggable data sources, engines, and output sinks.

This module provides a unified interface for running Neraium's structural intelligence
engine on any data source, with any engine configuration, and writing results to any
output destination.

Key components:
- DataSourceAdapter: Read frames from any source (CSV, Kafka, API, etc.)
- OutputSink: Write results to any destination (file, database, webhook, etc.)
- EngineConfig: Configure which engine components to enable
- AgnosticRunner: Orchestrate data flow through engine and outputs

Example usage:

    from neraium_core.agnostic_runner import AgnosticRunner
    from neraium_core.agnostic_runner.adapters import CsvDataSource, JsonFileSink

    runner = AgnosticRunner(
        data_source=CsvDataSource("data.csv"),
        output_sink=JsonFileSink("results.json"),
        mode="production"
    )
    metrics = runner.run()
"""

from .runner import AgnosticRunner, RunMetrics
from .adapters.registry import AdapterRegistry, get_registry, register_adapter
from .config import EngineConfig

__all__ = [
    "AgnosticRunner",
    "RunMetrics",
    "EngineConfig",
    "AdapterRegistry",
    "get_registry",
    "register_adapter",
]
