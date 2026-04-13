from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from neraium_core.data_connectors import (
    LiveConnectorError,
    MockLiveConnector,
    PolygonRESTConnector,
    create_stock_connector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--output", default="logs/live_stock_signals.csv")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--provider", default="polygon")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--provider-interval", default="1min")
    return parser.parse_args()


def _build_connector(args: argparse.Namespace):
    if getattr(args, "mock", False):
        return MockLiveConnector()
    if getattr(args, "provider", "polygon") == "polygon":
        key = args.api_key
        if not key:
            import os

            key = os.getenv("POLYGON_API_KEY")
        if key:
            return PolygonRESTConnector(api_key=key)
    return create_stock_connector(provider=getattr(args, "provider", "polygon"), api_key=getattr(args, "api_key", None), interval=getattr(args, "provider_interval", "1min"))


def _append_csv_row(output_path: str, row: dict[str, Any]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _validate_bar(bar: dict[str, Any]) -> tuple[dict[str, Any] | None, Exception | None]:
    required = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(bar.keys()))
    if missing:
        return None, ValueError(f"missing_fields: {','.join(missing)}")
    return bar, None


def _load_replay_bars(csv_path: str, tickers: list[str]) -> list[dict[str, Any]]:
    selected = {t.upper() for t in tickers}
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("ticker", "")).upper() not in selected:
                continue
            rows.append(row)
    rows.sort(key=lambda r: str(r.get("timestamp", "")))
    return rows


def _fetch_bars_per_symbol(connector: Any, tickers: list[str], cycle_started: datetime) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            fetched = connector.fetch_latest_bars([ticker])
        except LiveConnectorError:
            continue
        bars.extend(fetched)
    return bars


if __name__ == "__main__":
    parse_args()
