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
    LiveConnectorError,
    LiveMarketConnector,
    MockLiveConnector,
    create_stock_connector,
)
from neraium_core.intraday_output import compact_intraday_output
from neraium_core.live_runner import process_live_frame
from neraium_core.stock_market_adapter import build_stock_frame
from neraium_core.trading_signals import IntradaySignalPolicy, build_signal_decision

REQUIRED_BAR_FIELDS = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}


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
            "explanation_text": f"{state} regime from close-move score={score:.2f}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neraium against live/near-live stock bars (analytics only).")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument(
        "--provider",
        default=os.getenv("LIVE_DATA_PROVIDER", "polygon"),
        choices=["massive", "polygon", "alphavantage", "mock"],
    )
    parser.add_argument("--interval", "--poll-interval", dest="interval", type=float, default=float(os.getenv("LIVE_POLL_INTERVAL", "15")), help="Polling interval in seconds")
    parser.add_argument("--provider-interval", default=os.getenv("ALPHAVANTAGE_INTERVAL", "1min"), help="Alpha Vantage bar interval, e.g. 1min")
    parser.add_argument("--api-key", default=None, help="Provider API key override (else env var is used)")
    parser.add_argument("--fallback-to-mock", action="store_true", help="Use mock connector if live network/API calls fail")
    parser.add_argument("--mock", action="store_true", help="Shortcut for --provider mock")
    parser.add_argument("--output", "--output-log", dest="output", default=None, help="Optional CSV file to append live analytics rows")
    parser.add_argument("--max-iterations", type=int, default=0, help="Optional max polling iterations for smoke tests; 0 = run forever")
    parser.add_argument("--baseline-window", type=int, default=40)
    parser.add_argument("--recent-window", type=int, default=12)
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("INTRADAY_MIN_CONFIDENCE", "0.45")))
    parser.add_argument("--cooldown-seconds", type=float, default=float(os.getenv("INTRADAY_SIGNAL_COOLDOWN_SECONDS", "45")))
    parser.add_argument("--state-change-only", action="store_true")
    parser.add_argument("--warmup-bars", type=int, default=int(os.getenv("INTRADAY_WARMUP_BARS", "5")))
    parser.add_argument("--replay-csv", default=None, help="Replay historical bars from CSV instead of live provider")
    return parser.parse_args()


def _timestamp_to_epoch(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return float(ts.timestamp())


def _build_connector(args: argparse.Namespace) -> LiveMarketConnector:
    provider = "mock" if args.mock else args.provider
    api_key = args.api_key
    if provider in {"massive", "polygon"} and not api_key:
        api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
    elif provider == "alphavantage" and not api_key:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    return create_stock_connector(provider, api_key=api_key, interval=args.provider_interval)


def _append_csv_row(path: str, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "ticker",
        "state",
        "trading_signal",
        "confidence",
        "structural_drift_score",
        "latest_instability",
        "reason",
        "transition",
        "cooldown_remaining_seconds",
        "emitted",
    ]
    file_exists = output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _coerce_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _validate_bar(raw_bar: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    missing = sorted(name for name in REQUIRED_BAR_FIELDS if raw_bar.get(name) in (None, ""))
    if missing:
        return None, f"missing_fields:{','.join(missing)}"

    ts = _coerce_timestamp(raw_bar.get("timestamp"))
    if ts is None:
        return None, "invalid_timestamp"

    validated = dict(raw_bar)
    validated["timestamp"] = ts
    try:
        for key in ("open", "high", "low", "close", "volume"):
            validated[key] = float(validated[key])
    except (TypeError, ValueError):
        return None, "invalid_numeric_fields"

    if validated["high"] < validated["low"]:
        return None, "high_below_low"

    validated["ticker"] = str(validated["ticker"]).upper()
    validated["value"] = float(validated.get("value", validated["close"]))
    return validated, None


def _load_replay_bars(path: str, tickers: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            if str(raw.get("ticker", "")).upper() not in set(tickers):
                continue
            bar, err = _validate_bar(raw)
            if bar is not None:
                rows.append(bar)
            else:
                print(f"[REPLAY] skip malformed row for {raw.get('ticker')}: {err}")
    return sorted(rows, key=lambda item: (item["timestamp"], item["ticker"]))


def _fetch_bars_per_symbol(connector: LiveMarketConnector, tickers: list[str], cycle_started: datetime) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            symbol_bars = connector.fetch_latest_bars([ticker])
        except LiveConnectorError as exc:
            print(f"[{cycle_started.isoformat()}] live fetch error ticker={ticker}: {exc}")
            continue
        bars.extend(symbol_bars)
    return bars


def main() -> None:
    try:
        from neraium_core.alignment import StructuralEngine
    except ModuleNotFoundError:
        StructuralEngine = _FallbackStructuralEngine

    args = parse_args()
    tickers = [part.strip().upper() for part in args.tickers.split(",") if part.strip()]
    if not tickers:
        raise SystemExit("No tickers provided. Example: --tickers AAPL,MSFT")
    if args.interval <= 0:
        raise SystemExit("--interval must be > 0")

    print("[SAFETY MODE] Analytics/signals only. No brokerage execution is performed.")
    selected_provider = "mock" if args.mock else args.provider
    print(f"Provider={selected_provider} | Polling interval={args.interval}s | Tickers={','.join(tickers)}")

    policy = IntradaySignalPolicy(
        min_confidence=max(0.0, min(args.min_confidence, 1.0)),
        cooldown_seconds=max(0.0, args.cooldown_seconds),
        emit_on_state_change_only=bool(args.state_change_only),
        warmup_bars=max(1, int(args.warmup_bars)),
    )

    connector: LiveMarketConnector | None = None
    if not args.replay_csv:
        try:
            connector = _build_connector(args)
        except LiveConnectorError as exc:
            raise SystemExit(f"Connector setup failed: {exc}") from exc

    engines: dict[str, Any] = {
        ticker: StructuralEngine(baseline_window=args.baseline_window, recent_window=args.recent_window)
        for ticker in tickers
    }
    last_emitted: dict[str, dict[str, Any]] = {}
    last_seen_ts: dict[str, datetime] = {}
    bars_seen: dict[str, int] = {ticker: 0 for ticker in tickers}

    replay_rows = _load_replay_bars(args.replay_csv, tickers) if args.replay_csv else []
    replay_index = 0

    iteration = 0
    while True:
        iteration += 1
        cycle_started = datetime.now(timezone.utc)

        if args.replay_csv:
            if replay_index >= len(replay_rows):
                break
            current_ts = replay_rows[replay_index]["timestamp"]
            bars: list[dict[str, Any]] = []
            while replay_index < len(replay_rows) and replay_rows[replay_index]["timestamp"] == current_ts:
                bars.append(replay_rows[replay_index])
                replay_index += 1
        else:
            assert connector is not None
            bars = _fetch_bars_per_symbol(connector, tickers, cycle_started)
            if not bars and args.fallback_to_mock and not isinstance(connector, MockLiveConnector):
                print(f"[{cycle_started.isoformat()}] switching to mock connector fallback mode.")
                connector = MockLiveConnector()

        for raw_bar in bars:
            bar, bar_error = _validate_bar(raw_bar)
            if bar is None:
                print(f"[{cycle_started.isoformat()}] malformed bar dropped ({bar_error}): {raw_bar}")
                continue

            ticker = str(bar["ticker"]).upper()
            ts = bar["timestamp"]
            prev_ts = last_seen_ts.get(ticker)
            if prev_ts is not None and ts <= prev_ts:
                print(f"[{cycle_started.isoformat()}] duplicate_or_out_of_order_bar ticker={ticker} ts={ts.isoformat()} skipped")
                continue
            last_seen_ts[ticker] = ts

            frame = build_stock_frame(timestamp=_timestamp_to_epoch(ts), ticker=ticker, row_dict=bar)
            if not frame.get("sensor_values"):
                print(f"[{cycle_started.isoformat()}] empty sensor_values after frame conversion ticker={ticker}; bar dropped")
                continue

            try:
                result = process_live_frame(engines[ticker], frame)
            except Exception as exc:  # noqa: BLE001
                print(f"[{cycle_started.isoformat()}] engine processing failure ticker={ticker}: {exc}")
                continue

            result_dict = result if isinstance(result, dict) else {"result": result}
            bars_seen[ticker] += 1
            signal_decision = build_signal_decision(
                output=result_dict,
                ticker=ticker,
                timestamp=ts,
                policy=policy,
                previous=last_emitted.get(ticker),
                bars_seen=bars_seen[ticker],
            )
            row_out = compact_intraday_output(ticker=ticker, timestamp=ts, decision=signal_decision, result=result_dict)

            print(
                " | ".join(
                    [
                        row_out["timestamp"],
                        row_out["ticker"],
                        f"state={row_out['state']}",
                        f"signal={row_out['trading_signal']}",
                        f"conf={row_out['confidence']:.2f}",
                        f"emit={row_out['emitted']}",
                        f"reason={row_out['reason']}",
                    ]
                )
            )

            if row_out["emitted"]:
                last_emitted[ticker] = {
                    "timestamp": ts,
                    "trading_signal": row_out["trading_signal"],
                    "state": row_out["state"],
                    "structural_drift_score": row_out["structural_drift_score"],
                }
            if args.output:
                _append_csv_row(args.output, row_out)

        if args.max_iterations and iteration >= args.max_iterations:
            break
        if args.replay_csv:
            continue
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(0)
