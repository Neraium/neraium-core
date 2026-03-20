from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

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
        return [data]

    raise ValueError("Input JSON must be an object or array of objects")


def _pilot_view(result: dict[str, Any]) -> dict[str, Any]:
    return {k: result.get(k) for k in PILOT_KEYS}


def generate_signals(t: int, rng: np.random.Generator) -> dict[str, float]:
    """
    Evolving multivariate scenario (timestep index t >= 0).

    - t < 30: stable baseline — shared latent factor, tight coupling.
    - 30 <= t < 60: regime shift — one channel ramps steadily vs the others.
    - t >= 60: instability — independent high noise / coupling breakdown.
    """
    if t < 30:
        # Near-perfect coupling (same value on all channels) keeps drift low until t == 30.
        v = 10.0 + 0.02 * math.sin(float(t) / 4.0)
        return {"s1": v, "s2": v, "s3": v, "s4": v}

    if t < 60:
        # Steady increase on s1 breaks prior correlation geometry (regime shift).
        latent = rng.normal(0.0, 0.04)
        ramp = 0.095 * float(t - 30)
        base = 10.0
        eps = 0.02
        return {
            "s1": base + ramp + latent + rng.normal(0.0, eps),
            "s2": base + latent + rng.normal(0.0, eps),
            "s3": base + latent + rng.normal(0.0, eps),
            "s4": base + latent + rng.normal(0.0, eps),
        }

    # Coupling breakdown: weak shared structure, strong independent noise.
    u = float(t - 60)
    return {
        "s1": 8.0 + 0.08 * u + rng.normal(0.0, 1.4),
        "s2": 14.0 + rng.normal(0.0, 1.6),
        "s3": 6.0 + rng.normal(0.0, 1.5),
        "s4": 11.0 + rng.normal(0.0, 1.7),
    }


def _iso_timestamp(start: datetime, step: int) -> str:
    return (start + timedelta(seconds=step)).isoformat()


def _print_step_record(
    *,
    t: int,
    result: dict[str, Any],
) -> None:
    """One JSON object per line: timestamp, signals, score, status, interpreted_state."""
    pv = _pilot_view(result)
    signals = pv.get("signals")
    rounded_signals: dict[str, float | None] | None
    if isinstance(signals, dict):
        rounded_signals = {}
        for k, v in signals.items():
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                rounded_signals[str(k)] = None
            else:
                rounded_signals[str(k)] = round(float(v), 6)
    else:
        rounded_signals = None

    interpreted = result.get("interpreted_state")
    record: dict[str, Any] = {
        "t": t,
        "timestamp": pv.get("timestamp"),
        "signals": rounded_signals,
        "score": pv.get("score"),
        "status": pv.get("status"),
    }
    if interpreted is not None:
        record["interpreted_state"] = interpreted
    print(json.dumps(record, separators=(",", ":")))


def _run_scenario(
    *,
    timesteps: int,
    seed: int,
    baseline_window: int,
    recent_window: int,
) -> None:
    """Drive Neraium with generate_signals(t) for t in [0, timesteps)."""
    if timesteps < 100:
        raise ValueError("timesteps must be at least 100 for the pilot scenario")

    # Avoid per-step INFO logs flooding stdout when printing JSONL.
    for name in ("neraium_core.service", "neraium_core.store"):
        logging.getLogger(name).setLevel(logging.WARNING)

    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    tmp_dir = tempfile.mkdtemp(prefix="neraium_pilot_")
    try:
        tmp_path = Path(tmp_dir)
        engine = StructuralEngine(
            baseline_window=baseline_window,
            recent_window=recent_window,
            window_stride=1,
            regime_store_path=str(tmp_path / "regimes.json"),
        )
        store = ResultStore(db_path=str(tmp_path / "run.db"))
        service = StructuralMonitoringService(engine=engine, store=store)

        first_watch: int | None = None
        first_alert: int | None = None

        for t in range(timesteps):
            payload = {
                "timestamp": _iso_timestamp(start, t),
                "site_id": "pilot-scenario",
                "asset_id": "asset-1",
                "sensor_values": generate_signals(t, rng),
            }
            result = service.ingest_payload(payload)
            action = str(result.get("action_state", ""))
            if first_watch is None and action == "WATCH":
                first_watch = t
            if first_alert is None and action == "ALERT":
                first_alert = t

            _print_step_record(t=t, result=result)

        del service, store, engine
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Progression summary (operator-facing heuristic status)
    print(
        json.dumps(
            {
                "summary": {
                    "scenario": "stable → regime_shift → instability",
                    "phases": {"stable_t_lt_30": True, "regime_shift_30_le_t_lt_60": True, "instability_t_ge_60": True},
                    "first_WATCH_step": first_watch,
                    "first_ALERT_step": first_alert,
                    "note": "status is service action_state (STABLE/WATCH/ALERT); "
                    "interpreted_state is engine/decision-layer interpretation.",
                }
            },
            indent=2,
        ),
        flush=True,
    )


def _run_file_payloads(path: Path) -> None:
    """Original behavior: load JSON payload(s) and print pilot view."""
    payloads = _load_json_payloads(path)

    tmp_dir = tempfile.mkdtemp(prefix="neraium_pilot_")
    try:
        tmp_path = Path(tmp_dir)
        engine = StructuralEngine(
            baseline_window=5,
            recent_window=3,
            window_stride=1,
            regime_store_path=str(tmp_path / "regimes.json"),
        )
        store = ResultStore(db_path=str(tmp_path / "run.db"))
        service = StructuralMonitoringService(engine=engine, store=store)

        outputs: list[dict[str, Any]] = []
        for payload in payloads:
            result = service.ingest_payload(payload)
            outputs.append(_pilot_view(result))

        del service, store, engine
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if len(outputs) == 1:
        print(json.dumps(outputs[0], indent=2))
    else:
        print(json.dumps(outputs, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Neraium Pilot Hardening Mode: dynamic scenario or JSON payload file.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional JSON file: single payload, list, or {payloads|items|sequence: [...]}. "
        "If omitted, runs the built-in evolving scenario (≥100 steps).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable pilot debug logging (redacted).")
    parser.add_argument("--timesteps", type=int, default=120, help="Scenario length (default 120, min 100).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible scenario noise.")
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=24,
        help="StructuralEngine baseline window (scenario mode uses larger windows for drift).",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=8,
        help="StructuralEngine recent window (scenario mode).",
    )
    args = parser.parse_args()

    os.environ["NERAIUM_PILOT_HARDENING"] = "1"
    if args.debug:
        os.environ["NERAIUM_DEBUG_PILOT"] = "1"

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        _run_file_payloads(input_path)
        return

    _run_scenario(
        timesteps=args.timesteps,
        seed=args.seed,
        baseline_window=args.baseline_window,
        recent_window=args.recent_window,
    )


if __name__ == "__main__":
    main()
