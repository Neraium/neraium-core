from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


class LiveConnectorError(RuntimeError):
    """Raised when a live market connector cannot fetch or parse data."""


class LiveMarketConnector(ABC):
    """Abstract connector for fetching latest market bars."""

    @abstractmethod
    def fetch_latest_bars(self, tickers: list[str]) -> list[dict[str, Any]]:
        """Return latest bars normalized to canonical keys."""


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Normalize record values into float-only dictionaries."""
    normalized: List[Dict[str, float]] = []
    for row in records:
        normalized.append({str(k): float(v) for k, v in row.items()})
    return normalized


class AlphaVantageRESTConnector(LiveMarketConnector):
    """Fetch latest intraday bars from Alpha Vantage REST API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        interval: str = "1min",
        function: str = "TIME_SERIES_INTRADAY",
        timeout_seconds: int = 15,
    ) -> None:
        self.api_key = (api_key or os.getenv("ALPHAVANTAGE_API_KEY", "")).strip()
        if not self.api_key:
            raise LiveConnectorError(
                "Missing API key. Set ALPHAVANTAGE_API_KEY or pass --api-key for provider=alphavantage."
            )
        self.interval = interval
        self.function = function
        self.timeout_seconds = timeout_seconds

    def _fetch_symbol_payload(self, symbol: str) -> dict[str, Any]:
        params = {
            "function": self.function,
            "symbol": symbol,
            "interval": self.interval,
            "apikey": self.api_key,
            "outputsize": "compact",
            "datatype": "json",
        }
        url = f"https://www.alphavantage.co/query?{urlencode(params)}"

        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise LiveConnectorError(f"Network error while requesting {symbol}: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LiveConnectorError(f"Invalid JSON response for {symbol}") from exc

        if "Note" in data:
            raise LiveConnectorError(
                f"Provider rate-limit notice for {symbol}: {data['Note']}"
            )
        if "Error Message" in data:
            raise LiveConnectorError(
                f"Provider rejected symbol {symbol}: {data['Error Message']}"
            )
        return data

    def _extract_latest_bar(self, symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
        key = f"Time Series ({self.interval})"
        series = payload.get(key)
        if not isinstance(series, dict) or not series:
            raise LiveConnectorError(
                f"No intraday series found for {symbol}. Check interval ({self.interval}) and API plan support."
            )

        latest_ts = max(series.keys())
        latest = series[latest_ts]

        try:
            open_px = float(latest.get("1. open"))
            high_px = float(latest.get("2. high"))
            low_px = float(latest.get("3. low"))
            close_px = float(latest.get("4. close"))
            volume = float(latest.get("5. volume"))
        except (TypeError, ValueError) as exc:
            raise LiveConnectorError(f"Malformed OHLCV payload for {symbol} at {latest_ts}") from exc

        ts = datetime.fromisoformat(str(latest_ts)).replace(tzinfo=timezone.utc)
        return {
            "timestamp": ts,
            "ticker": symbol,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": volume,
            "value": close_px,
        }

    def fetch_latest_bars(self, tickers: list[str]) -> list[dict[str, Any]]:
        bars: list[dict[str, Any]] = []
        for symbol in tickers:
            payload = self._fetch_symbol_payload(symbol)
            bars.append(self._extract_latest_bar(symbol=symbol, payload=payload))
        return bars


class PolygonRESTConnector(LiveMarketConnector):
    """Fetch latest minute bars from Polygon REST API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.api_key = (api_key or os.getenv("POLYGON_API_KEY", "")).strip()
        if not self.api_key:
            raise LiveConnectorError(
                "Missing API key. Set POLYGON_API_KEY or pass --api-key for provider=polygon."
            )
        self.timeout_seconds = timeout_seconds

    def _fetch_symbol_payload(self, symbol: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        date_to = now.date().isoformat()
        date_from = (now - timedelta(days=5)).date().isoformat()
        params = {
            "adjusted": "true",
            "sort": "desc",
            "limit": "1",
            "apiKey": self.api_key,
        }
        url = (
            "https://api.polygon.io/v2/aggs/ticker/"
            f"{symbol}/range/1/minute/{date_from}/{date_to}?{urlencode(params)}"
        )

        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise LiveConnectorError(f"Network error while requesting {symbol}: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LiveConnectorError(f"Invalid JSON response for {symbol}") from exc

        status = str(data.get("status", "")).upper()
        if status not in {"OK", "DELAYED"}:
            err = data.get("error") or data.get("message") or "unknown provider error"
            raise LiveConnectorError(f"Provider rejected symbol {symbol}: {err}")
        return data

    def _extract_latest_bar(self, symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            raise LiveConnectorError(f"No bar data found for {symbol}")
        latest = results[0]

        try:
            open_px = float(latest.get("o"))
            high_px = float(latest.get("h"))
            low_px = float(latest.get("l"))
            close_px = float(latest.get("c"))
            volume = float(latest.get("v"))
            ts_ms = int(latest.get("t"))
        except (TypeError, ValueError) as exc:
            raise LiveConnectorError(f"Malformed OHLCV payload for {symbol}") from exc

        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        return {
            "timestamp": ts,
            "ticker": symbol,
            "open": open_px,
            "high": high_px,
            "low": low_px,
            "close": close_px,
            "volume": volume,
            "value": close_px,
        }

    def fetch_latest_bars(self, tickers: list[str]) -> list[dict[str, Any]]:
        bars: list[dict[str, Any]] = []
        for symbol in tickers:
            payload = self._fetch_symbol_payload(symbol)
            bars.append(self._extract_latest_bar(symbol=symbol, payload=payload))
        return bars


class MockLiveConnector(LiveMarketConnector):
    """Local deterministic connector used for smoke tests and offline development."""

    def __init__(self, base_price: float = 100.0) -> None:
        self.base_price = base_price
        self._step = 0

    def fetch_latest_bars(self, tickers: list[str]) -> list[dict[str, Any]]:
        self._step += 1
        ts = datetime.now(timezone.utc)
        bars: list[dict[str, Any]] = []
        for idx, ticker in enumerate(tickers):
            drift = (self._step + idx) * 0.05
            open_px = self.base_price + drift
            close_px = open_px + ((-1) ** self._step) * 0.2
            high_px = max(open_px, close_px) + 0.1
            low_px = min(open_px, close_px) - 0.1
            volume = 1_000 + self._step * 10 + idx
            bars.append(
                {
                    "timestamp": ts,
                    "ticker": ticker,
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": float(volume),
                    "value": close_px,
                }
            )
        return bars


__all__ = [
    "normalize_records",
    "LiveMarketConnector",
    "LiveConnectorError",
    "AlphaVantageRESTConnector",
    "PolygonRESTConnector",
    "MockLiveConnector",
    "create_stock_connector",
]


def create_stock_connector(
    provider: str,
    *,
    api_key: str | None = None,
    interval: str = "1min",
    timeout_seconds: int = 15,
) -> LiveMarketConnector:
    """Build a stock-market connector from a simple provider name/env setup."""
    normalized_provider = str(provider).strip().lower()
    if normalized_provider == "mock":
        return MockLiveConnector()
    if normalized_provider == "polygon":
        return PolygonRESTConnector(api_key=api_key, timeout_seconds=timeout_seconds)
    if normalized_provider == "alphavantage":
        return AlphaVantageRESTConnector(
            api_key=api_key,
            interval=interval,
            timeout_seconds=timeout_seconds,
        )
    raise LiveConnectorError(
        f"Unsupported provider {provider!r}. Supported values: polygon, alphavantage, mock."
    )
