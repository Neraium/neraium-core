from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


PILOT_KEYS = ("timestamp", "signals", "score", "status", "aligned", "anomaly")


def _load_json_payloads(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("payloads", "items", "sequence"):
            maybe = data.get(key)
            if isinstance(maybe, list):
                return maybe
        # Single payload object
        return [data]

    raise ValueError("Input JSON must be an object or array of objects")


def _pilot_view(result: dict[str, Any]) -> dict[str, Any]:
    return {k: result.get(k) for k in PILOT_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neraium Pilot Hardening Mode on sample JSON input.")
    parser.add_argument("--input", required=True, help="Path to a JSON file containing a payload or payload sequence.")
    parser.add_argument("--debug", action="store_true", help="Enable pilot debug logging (redacted).")
    parser.add_argument("--baseline-window", type=int, default=5, help="StructuralEngine baseline window size.")
    parser.add_argument("--recent-window", type=int, default=3, help="StructuralEngine recent window size.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    os.environ["NERAIUM_PILOT_HARDENING"] = "1"
    if args.debug:
        os.environ["NERAIUM_DEBUG_PILOT"] = "1"

    payloads = _load_json_payloads(input_path)

    # Keep any engine/store artifacts isolated for the run.
    # Use mkdtemp + best-effort rmtree: on Windows, SQLite WAL can briefly lock `run.db`
    # after connections close; `TemporaryDirectory` cleanup can raise PermissionError.
    tmp_dir = tempfile.mkdtemp(prefix="neraium_pilot_")
    try:
        tmp_path = Path(tmp_dir)
        engine = StructuralEngine(
            baseline_window=args.baseline_window,
            recent_window=args.recent_window,
            window_stride=1,
            regime_store_path=str(tmp_path / "regimes.json"),
        )
        store = ResultStore(db_path=str(tmp_path / "run.db"))
        service = StructuralMonitoringService(engine=engine, store=store)

        outputs: list[dict[str, Any]] = []
        for payload in payloads:
            result = service.ingest_payload(payload)
            outputs.append(_pilot_view(result))

        # Drop references so SQLite releases file handles before temp dir removal (Windows).
        del service, store, engine
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print structured pilot view (never raw sensor values outside the `signals` field).
    if len(outputs) == 1:
        print(json.dumps(outputs[0], indent=2))
    else:
        print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

