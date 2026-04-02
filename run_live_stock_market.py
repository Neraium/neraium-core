from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neraium_core.data_connectors import (
    AlphaVantageRESTConnector,
    LiveConnectorError,
    LiveMarketConnector,
    MockLiveConnector,
    PolygonRESTConnector,
)
from neraium_core.live_runner import process_live_frame
from neraium_core.stock_market_adapter import build_stock_frame
from neraium_core.trading_signals import map_structural_output_to_signal


class _FallbackStructuralEngine:
    def __init__(self, baseline_window: int = 40, recent_window: int = 12) -> None:
        self.baseline_window = baseline_window
        self.recent_window = recent_window
        self._last_value: float | None = None

    def process_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        sensors = frame.get("sensor_values", {})
        current = sensors.get("close", sensors.get("value", 0.0))
        change = 0.0 if self._last_value in {None, 0.0} else (float(current) - float(self._last_value)) / float(self._last_value)
        self._last_value = float(current)

        score = min(3.5, abs(change) * 100.0)
        if score >= 3.0:
            state = "ALERT"
        elif score >= 2.0:
            state = "WATCH"
        else:
            state = "NORMAL"

        return {
            "classification": state,
            "drift_state": state,
            "structural_drift_score": score,
            "latest_instability": score,
            "system_health": max(0.0, 100.0 - score * 20.0),
            "evidence_confidence": 0.75,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neraium against live/near-live stock bars (analytics only).")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument("--provider", default=os.getenv("LIVE_DATA_PROVIDER", "polygon"), choices=["polygon", "alphavantage", "mock"])
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("LIVE_POLL_INTERVAL", "15")), help="Polling interval in seconds")
    parser.add_argument("--provider-interval", default=os.getenv("ALPHAVANTAGE_INTERVAL", "1min"), help="Alpha Vantage bar interval, e.g. 1min")
    parser.add_argument("--api-key", default=None, help="Provider API key override (else env var is used)")
    parser.add_argument("--fallback-to-mock", action="store_true", help="Use mock connector if live network/API calls fail")
    parser.add_argument("--output-log", default=None, help="Optional CSV file to append live analytics rows")
    parser.add_argument("--max-iterations", type=int, default=0, help="Optional max polling iterations for smoke tests; 0 = run forever")
    parser.add_argument("--baseline-window", type=int, default=40)
    parser.add_argument("--recent-window", type=int, default=12)
    return parser.parse_args()


def _timestamp_to_epoch(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return float(ts.timestamp())


def _build_connector(args: argparse.Namespace) -> LiveMarketConnector:
    if args.provider == "mock":
        return MockLiveConnector()
    if args.provider == "polygon":
        return PolygonRESTConnector(api_key=args.api_key or os.getenv("POLYGON_API_KEY"))
    if args.provider == "alphavantage":
        return AlphaVantageRESTConnector(
            api_key=args.api_key or os.getenv("ALPHAVANTAGE_API_KEY"),
            interval=args.provider_interval,
        )
    raise LiveConnectorError(f"Unsupported provider: {args.provider}")


def _append_csv_row(path: str, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "ticker",
        "state",
        "trading_signal",
        "structural_drift_score",
        "latest_instability",
        "system_health",
        "evidence_confidence",
    ]
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _extract_state(result: dict[str, Any]) -> str:
    return str(result.get("drift_state", result.get("classification", "UNKNOWN")))


def main() -> None:
    try:
        from neraium_core.alignment import StructuralEngine
    except ModuleNotFoundError:
        StructuralEngine = _FallbackStructuralEngine

    args = parse_args()
    tickers = [part.strip().upper() for part in args.tickers.split(",") if part.strip()]
    if not tickers:
        raise SystemExit("No tickers provided. Example: --tickers AAPL,MSFT")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be > 0")

    print("[SAFETY MODE] Analytics/signals only. No brokerage execution is performed.")
    print(f"Provider={args.provider} | Polling interval={args.poll_interval}s | Tickers={','.join(tickers)}")

    try:
        connector = _build_connector(args)
    except LiveConnectorError as exc:
        raise SystemExit(f"Connector setup failed: {exc}") from exc

    engines: dict[str, Any] = {
        ticker: StructuralEngine(baseline_window=args.baseline_window, recent_window=args.recent_window)
        for ticker in tickers
    }

    iteration = 0
    while True:
        iteration += 1
        cycle_started = datetime.now(timezone.utc)

        try:
            bars = connector.fetch_latest_bars(tickers)
        except LiveConnectorError as exc:
            print(f"[{cycle_started.isoformat()}] live fetch error: {exc}")
            if args.fallback_to_mock and not isinstance(connector, MockLiveConnector):
                print(f"[{cycle_started.isoformat()}] switching to mock connector fallback mode.")
                connector = MockLiveConnector()
            if args.max_iterations and iteration >= args.max_iterations:
                break
            time.sleep(args.poll_interval)
            continue

        for bar in bars:
            ticker = str(bar["ticker"]).upper()
            ts = bar["timestamp"]
            if not isinstance(ts, datetime):
                raise SystemExit("Connector returned invalid timestamp type; expected datetime")

            frame = build_stock_frame(
                timestamp=_timestamp_to_epoch(ts),
                ticker=ticker,
                row_dict=bar,
            )
            result = process_live_frame(engines[ticker], frame)
            result_dict = result if isinstance(result, dict) else {"result": result}

            row_out = {
                "timestamp": ts.isoformat(),
                "ticker": ticker,
                "state": _extract_state(result_dict),
                "trading_signal": map_structural_output_to_signal(result_dict),
                "structural_drift_score": result_dict.get("structural_drift_score"),
                "latest_instability": result_dict.get("latest_instability"),
                "system_health": result_dict.get("system_health"),
                "evidence_confidence": result_dict.get("evidence_confidence"),
            }
            print(
                " | ".join(
                    [
                        row_out["timestamp"],
                        row_out["ticker"],
                        f"state={row_out['state']}",
                        f"signal={row_out['trading_signal']}",
                        f"drift={row_out['structural_drift_score']}",
                        f"instability={row_out['latest_instability']}",
                        f"health={row_out['system_health']}",
                    ]
                )
            )

            if args.output_log:
                _append_csv_row(args.output_log, row_out)

        if args.max_iterations and iteration >= args.max_iterations:
            break
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
