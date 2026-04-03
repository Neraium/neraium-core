import json
from typing import Any

from neraium_core.data_connectors import (
    LiveConnectorError,
    MassiveRESTConnector,
    PolygonRESTConnector,
    create_stock_connector,
    normalize_records,
)


def test_normalize_records_casts_values_to_float() -> None:
    rows = [{"cpu": 1, "mem": "2.5"}, {"cpu": 3.2, "mem": 4}]

    out = normalize_records(rows)

    assert out == [{"cpu": 1.0, "mem": 2.5}, {"cpu": 3.2, "mem": 4.0}]


def test_polygon_connector_extracts_latest_bar(monkeypatch: Any) -> None:
    payload = {
        "status": "OK",
        "results": [
            {"o": 100.0, "h": 101.0, "l": 99.5, "c": 100.7, "v": 12345, "t": 1712059200000}
        ],
    }

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(url: str, timeout: int = 15) -> _Response:
        assert "api.polygon.io" in url
        assert timeout == 15
        return _Response()

    monkeypatch.setattr("neraium_core.data_connectors.urlopen", _fake_urlopen)
    connector = PolygonRESTConnector(api_key="test-key")

    bars = connector.fetch_latest_bars(["AAPL"])
    assert len(bars) == 1
    assert bars[0]["ticker"] == "AAPL"
    assert bars[0]["open"] == 100.0
    assert bars[0]["close"] == 100.7
    assert bars[0]["volume"] == 12345.0
    assert bars[0]["value"] == 100.7


def test_create_stock_connector_accepts_massive_alias() -> None:
    connector = create_stock_connector("massive", api_key="test-key")
    assert isinstance(connector, MassiveRESTConnector)


def test_polygon_connector_uses_massive_api_key_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "massive-env-key")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    connector = PolygonRESTConnector()
    assert connector.api_key == "massive-env-key"


def test_polygon_connector_requires_massive_or_polygon_api_key(monkeypatch: Any) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    try:
        PolygonRESTConnector()
    except LiveConnectorError as exc:
        assert "MASSIVE_API_KEY" in str(exc)
        assert "POLYGON_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected LiveConnectorError when no API key env vars are set")
