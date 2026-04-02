from __future__ import annotations

import argparse

import pandas as pd

from neraium_core.market_data_loader import load_market_data
from neraium_core.stock_market_adapter import build_stock_frame
from neraium_core.trading_signals import map_structural_output_to_signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neraium StructuralEngine across market CSV rows.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--ticker-column", default=None, help="Optional ticker column override")
    parser.add_argument("--timestamp-column", default=None, help="Optional timestamp column override")
    parser.add_argument("--baseline-window", type=int, default=40)
    parser.add_argument("--recent-window", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    from neraium_core.alignment import StructuralEngine

    args = parse_args()
    df = load_market_data(
        args.input,
        timestamp_column=args.timestamp_column,
        ticker_column=args.ticker_column,
    )

    engines: dict[str, StructuralEngine] = {}
    outputs: list[dict] = []

    for row in df.to_dict(orient="records"):
        ticker = str(row["ticker"])
        timestamp = row["timestamp"].timestamp()
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

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
