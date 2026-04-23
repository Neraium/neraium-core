"""Day 13: connectors and registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from neraium.connectors.csv_connector import CSVConnector
from neraium.connectors.mock_api_connector import MockAPIConnector
from neraium.connectors.registry import get_connector
from neraium.settings import load_settings, with_overrides


def test_csv_connector_loads_asset() -> None:
    root = Path(__file__).resolve().parents[1]
    c = CSVConnector(root / "sample_data")
    df = c.load_asset_data("spy")
    assert not df.empty
    assert "close" in [x.lower() for x in df.columns]


def test_mock_api_connector_loads_asset() -> None:
    root = Path(__file__).resolve().parents[1]
    fx = root / "tests" / "fixtures" / "mock_api"
    c = MockAPIConnector(fx, simulated_latency_ms=0.0)
    df = c.load_asset_data("spy")
    assert not df.empty
    meta = c.get_source_metadata()
    assert meta["connector"] == "MockAPIConnector"
    assert "simulated_latency_ms" in meta


def test_registry_returns_connector() -> None:
    s = with_overrides(load_settings("dev"), source_type="csv")
    c1 = get_connector("csv", s)
    assert c1.get_name() == "csv"

    s2 = with_overrides(
        load_settings("dev"),
        source_type="mock_api",
        mock_api_fixture_dir=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mock_api",
    )
    c2 = get_connector("mock_api", s2)
    assert c2.get_name() == "mock_api"


def test_registry_unknown() -> None:
    s = load_settings("dev")
    with pytest.raises(ValueError, match="Unsupported"):
        get_connector("postgres", s)
