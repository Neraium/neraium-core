from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - fallback runtime path
    pd = None

from neraium_core.market_data_loader import load_market_data
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
    parser = argparse.ArgumentParser(description="Run Neraium StructuralEngine across market CSV rows.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--ticker-column", default=None, help="Optional ticker column override")
    parser.add_argument("--timestamp-column", default=None, help="Optional timestamp column override")
    parser.add_argument("--baseline-window", type=int, default=40)
    parser.add_argument("--recent-window", type=int, default=12)
    return parser.parse_args()


def _timestamp_to_epoch(value: Any) -> float:
    if hasattr(value, "timestamp"):
        return float(value.timestamp())
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return float(parsed.timestamp())


def _iter_rows(dataset: Any) -> list[dict[str, Any]]:
    if hasattr(dataset, "to_dict"):
        return dataset.to_dict(orient="records")
    return list(dataset)


def _write_rows_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = {
                k: (v.isoformat() if isinstance(v, datetime) else v)
                for k, v in row.items()
            }
            writer.writerow(serialized)


def main() -> None:
    try:
        from neraium_core.alignment import StructuralEngine
    except ModuleNotFoundError:
        StructuralEngine = _FallbackStructuralEngine

    args = parse_args()
    df = load_market_data(
        args.input,
        timestamp_column=args.timestamp_column,
        ticker_column=args.ticker_column,
    )

    engines: dict[str, Any] = {}
    outputs: list[dict[str, Any]] = []

    for row in _iter_rows(df):
        ticker = str(row["ticker"])
        timestamp = _timestamp_to_epoch(row["timestamp"])
        engine = engines.get(ticker)
        if engine is None:
            engine = StructuralEngine(baseline_window=args.baseline_window, recent_window=args.recent_window)
            engines[ticker] = engine

        frame = build_stock_frame(timestamp=timestamp, ticker=ticker, row_dict=row)
        result = engine.process_frame(frame)

        merged = dict(row)
        merged.update(result if isinstance(result, dict) else {"result": result})
        merged["trading_signal"] = map_structural_output_to_signal(result if isinstance(result, dict) else {})
        outputs.append(merged)

    if pd is not None:
        out_df = pd.DataFrame(outputs)
        out_df.to_csv(args.output, index=False)
    else:
        _write_rows_csv(args.output, outputs)


if __name__ == "__main__":
    main()
