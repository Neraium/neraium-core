from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from neraium_core.markets.data.historical_ingest import HistoricalIngestService
from neraium_core.markets.evidence.evidence_log import EvidenceLog
from neraium_core.markets.persistence.sqlite_store import MarketsSQLiteStore
from neraium_core.markets.replay import run_signal_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="SPY,QQQ,AAPL,NVDA")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    ingest = HistoricalIngestService()
    fetch = ingest.fetch_massive(symbols=symbols, timeframe=args.timeframe, start_date=args.start, end_date=args.end)
    dataset_path = Path(fetch["dataset"]["dataset_path"])

    run_id = f"massive_replay_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_csv = Path("artifacts/neraium_markets/replay") / f"{run_id}.csv"
    evidence = EvidenceLog(path=Path("artifacts/neraium_markets/replay") / f"{run_id}.jsonl")
    rows = run_signal_replay(dataset_path, evidence, timeframe=args.timeframe, csv_output_path=out_csv, symbols=symbols)

    store = MarketsSQLiteStore()
    store.record_replay_run(run_id, {"timeframe": args.timeframe, "symbols": symbols, "dataset_path": str(dataset_path), "csv_path": str(out_csv)})
    for row in rows:
        store.record_replay_output(run_id, row)
    print({"run_id": run_id, "rows": len(rows), "csv": str(out_csv)})


if __name__ == "__main__":
    main()
