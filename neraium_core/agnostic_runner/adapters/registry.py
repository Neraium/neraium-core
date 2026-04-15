"""Registry for dynamically discovering and loading adapters."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Type

from .base import DataSourceAdapter, OutputSink

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Central registry for data source and output sink adapters."""

    def __init__(self) -> None:
        self._sources: dict[str, Type[DataSourceAdapter]] = {}
        self._sinks: dict[str, Type[OutputSink]] = {}
        self._source_factories: dict[str, Callable] = {}
        self._sink_factories: dict[str, Callable] = {}

    def register_source(
        self, name: str, adapter_class: Type[DataSourceAdapter], factory: Callable | None = None
    ) -> None:
        """Register a data source adapter."""
        self._sources[name] = adapter_class
        if factory:
            self._source_factories[name] = factory
        logger.debug(f"Registered data source adapter: {name}")

    def register_sink(
        self, name: str, adapter_class: Type[OutputSink], factory: Callable | None = None
    ) -> None:
        """Register an output sink adapter."""
        self._sinks[name] = adapter_class
        if factory:
            self._sink_factories[name] = factory
        logger.debug(f"Registered output sink adapter: {name}")

    def get_source(self, name: str) -> Type[DataSourceAdapter]:
        """Get a data source adapter class by name."""
        if name not in self._sources:
            raise ValueError(f"Unknown data source adapter: {name}. Available: {list(self._sources.keys())}")
        return self._sources[name]

    def get_sink(self, name: str) -> Type[OutputSink]:
        """Get an output sink adapter class by name."""
        if name not in self._sinks:
            raise ValueError(f"Unknown output sink adapter: {name}. Available: {list(self._sinks.keys())}")
        return self._sinks[name]

    def create_source(self, name: str, config: dict[str, Any]) -> DataSourceAdapter:
        """Create a data source adapter instance."""
        if name in self._source_factories:
            return self._source_factories[name](config)
        adapter_class = self.get_source(name)
        return adapter_class(config)

    def create_sink(self, name: str, config: dict[str, Any]) -> OutputSink:
        """Create an output sink adapter instance."""
        if name in self._sink_factories:
            return self._sink_factories[name](config)
        adapter_class = self.get_sink(name)
        return adapter_class(config)

    def list_sources(self) -> list[str]:
        """List all registered data source adapters."""
        return sorted(self._sources.keys())

    def list_sinks(self) -> list[str]:
        """List all registered output sink adapters."""
        return sorted(self._sinks.keys())

    def auto_discover(self, package_path: str = "neraium_core.agnostic_runner.adapters") -> None:
        """Auto-discover and register adapters from module."""
        try:
            module = importlib.import_module(package_path)
            # Import known adapter modules to trigger registrations
            for adapter_module in [
                "csv_source",
                "json_sink",
                "kafka_source",
                "kafka_sink",
                "api_source",
                "database_sink",
                "webhook_sink",
            ]:
                try:
                    importlib.import_module(f"{package_path}.{adapter_module}")
                except ImportError:
                    logger.debug(f"Adapter module not available: {adapter_module}")
        except ImportError as e:
            logger.warning(f"Could not auto-discover adapters: {e}")


# Global registry instance
_global_registry: AdapterRegistry | None = None


def get_registry() -> AdapterRegistry:
    """Get or create the global adapter registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AdapterRegistry()
        _register_builtin_adapters(_global_registry)
    return _global_registry


def register_adapter(adapter_type: str, name: str, adapter_class: Type, factory: Callable | None = None) -> None:
    """Register an adapter globally."""
    registry = get_registry()
    if adapter_type == "source":
        registry.register_source(name, adapter_class, factory)
    elif adapter_type == "sink":
        registry.register_sink(name, adapter_class, factory)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


def _register_builtin_adapters(registry: AdapterRegistry) -> None:
    """Register built-in adapters."""
    from .csv_source import CsvDataSource
    from .json_sink import JsonFileSink, JsonLineSink

    registry.register_source("csv", CsvDataSource)
    registry.register_sink("json", JsonFileSink)
    registry.register_sink("jsonl", JsonLineSink)

    # Optional adapters (may not have dependencies installed)
    try:
        from .kafka_source import KafkaDataSource
        from .kafka_sink import KafkaSink
        registry.register_source("kafka", KafkaDataSource)
        registry.register_sink("kafka", KafkaSink)
    except ImportError:
        logger.debug("Kafka adapters not available (kafka-python not installed)")

    try:
        from .api_source import ApiStreamSource
        registry.register_source("api", ApiStreamSource)
    except ImportError:
        logger.debug("API source adapter not available")

    try:
        from .database_sink import DatabaseSink
        registry.register_sink("database", DatabaseSink)
    except ImportError:
        logger.debug("Database sink adapter not available (sqlalchemy not installed)")

    try:
        from .webhook_sink import WebhookSink
        registry.register_sink("webhook", WebhookSink)
    except ImportError:
        logger.debug("Webhook sink adapter not available (requests not installed)")
