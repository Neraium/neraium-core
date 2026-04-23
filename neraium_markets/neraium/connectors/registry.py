"""Day 13: resolve connectors from settings (no hardcoded imports at call sites)."""

from __future__ import annotations

from neraium.connectors.base import MarketDataConnector
from neraium.connectors.csv_connector import CSVConnector
from neraium.connectors.mock_api_connector import MockAPIConnector
from neraium.settings import Settings


def get_connector(source_type: str, settings: Settings) -> MarketDataConnector:
    """Return a connector instance for ``source_type`` (``csv`` | ``mock_api``)."""
    st = source_type.strip().lower()
    if st == "csv":
        return CSVConnector(settings.data_dir, source_label="csv")
    if st == "mock_api":
        fx = settings.mock_api_fixture_dir or settings.data_dir
        return MockAPIConnector(
            fx,
            simulated_latency_ms=settings.mock_api_simulated_latency_ms,
            source_name="mock_api",
        )
    raise ValueError(f"Unsupported source_type: {source_type!r}; expected 'csv' or 'mock_api'")
