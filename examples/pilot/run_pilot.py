from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neraium_core.alignment import StructuralEngine
from neraium_core.service import StructuralMonitoringService
from neraium_core.store import ResultStore


PILOT_KEYS = ("timestamp", "signals", "score", "status", "aligned", "anomaly")

# Scenario-only: degradation logs to stderr (JSONL stays on stdout).
SCENARIO_LOG = logging.getLogger("neraium_pilot.scenario")


def _setup_scenario_logging() -> None:
    SCENARIO_LOG.setLevel(logging.INFO)
    if not SCENARIO_LOG.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] neraium_pilot.scenario: %(message)s"))
        SCENARIO_LOG.addHandler(handler)
    SCENARIO_LOG.propagate = False


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


def _scenario_timestamp(start: datetime, logical_t: int) -> tuple[str, list[str]]:
    """
    Nominal one-second cadence, except a window of delayed timestamps (non-monotonic vs ingest order).
    """
    tags: list[str] = []
    if 70 <= logical_t < 80:
        # Timestamp lags logical time by 10 steps (arrives / recorded late).
        sec = max(0, logical_t - 10)
        tags.append("delayed_timestamp_lag_10_steps")
        return (start + timedelta(seconds=sec)).isoformat(), tags
    return _iso_timestamp(start, logical_t), tags


def _rounded_signals_from_result(result: dict[str, Any]) -> dict[str, float | None] | None:
    pv = _pilot_view(result)
    signals = pv.get("signals")
    if not isinstance(signals, dict):
        return None
    rounded: dict[str, float | None] = {}
    for k, v in signals.items():
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            rounded[str(k)] = None
        else:
            rounded[str(k)] = round(float(v), 6)
    return rounded


def _result_score(result: dict[str, Any]) -> float | None:
    pv = _pilot_view(result)
    s = pv.get("score")
    if s is None:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _save_result_row(*, timestep: int, result: dict[str, Any]) -> dict[str, Any]:
    """Compact row for pilot_results.json (requested schema)."""
    return {
        "timestep": timestep,
        "signals": _rounded_signals_from_result(result),
        "state": result.get("state"),
        "interpreted_state": result.get("interpreted_state"),
        "score": _result_score(result),
    }


def _write_pilot_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    SCENARIO_LOG.info("saved %s rows to %s", len(rows), path.resolve())


def _apply_scenario_degradations(
    sensors: dict[str, float],
    logical_t: int,
) -> tuple[dict[str, float | None], list[str]]:
    """
    Inject missing data, flatline, etc. Pilot pipeline accepts None for missing sensors.
    """
    tags: list[str] = []
    out: dict[str, float | None] = dict(sensors)

    if 40 <= logical_t < 45:
        out["s2"] = None
        out["s3"] = None
        tags.append("missing_s2_s3")

    if logical_t >= 75:
        # Partial sensor failure: s4 stuck (same value every frame after onset).
        out["s4"] = 9.5
        tags.append("s4_flatline")

    return out, tags


def _print_step_record(
    *,
    ingest_seq: int,
    logical_t: int,
    result: dict[str, Any],
    degraded: list[str] | None = None,
    save_rows: list[dict[str, Any]] | None = None,
) -> None:
    """One JSON object per line: pilot fields + engine state + interpreted_state + confidence."""
    pv = _pilot_view(result)
    rounded_signals = _rounded_signals_from_result(result)

    record: dict[str, Any] = {
        "ingest_seq": ingest_seq,
        "logical_t": logical_t,
        "timestamp": pv.get("timestamp"),
        "signals": rounded_signals,
        "score": pv.get("score"),
        "status": pv.get("status"),
        "state": result.get("state"),
        "interpreted_state": result.get("interpreted_state"),
        "confidence": result.get("confidence"),
    }
    if degraded:
        record["degraded"] = degraded
    print(json.dumps(record, separators=(",", ":")))

    if save_rows is not None:
        save_rows.append(_save_result_row(timestep=logical_t, result=result))


def _log_engine_data_quality_if_degraded(
    *,
    logical_t: int,
    ingest_seq: int,
    result: dict[str, Any],
) -> None:
    dq = result.get("data_quality_summary")
    if not isinstance(dq, dict):
        return
    # Only log when the gate explicitly failed (not missing/unknown).
    if dq.get("gate_passed") is not False:
        return
    SCENARIO_LOG.info(
        "engine_data_quality_degraded ingest_seq=%s logical_t=%s gate_passed=False missingness_rate=%s statuses=%s",
        ingest_seq,
        logical_t,
        dq.get("missingness_rate"),
        dq.get("statuses"),
    )


def _run_scenario(
    *,
    timesteps: int,
    seed: int,
    baseline_window: int,
    recent_window: int,
    results_path: Path,
) -> None:
    """Drive Neraium with generate_signals(t) for t in [0, timesteps)."""
    if timesteps < 100:
        raise ValueError("timesteps must be at least 100 for the pilot scenario")

    _setup_scenario_logging()

    # Avoid per-step INFO logs flooding stdout when printing JSONL.
    for name in ("neraium_core.service", "neraium_core.store"):
        logging.getLogger(name).setLevel(logging.WARNING)

    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    save_rows: list[dict[str, Any]] = []
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
        ingest_seq = 0

        for logical_t in range(timesteps):
            raw = generate_signals(logical_t, rng)
            sensors, deg_tags = _apply_scenario_degradations(raw, logical_t)
            ts, ts_tags = _scenario_timestamp(start, logical_t)
            step_tags = list(deg_tags) + list(ts_tags)

            if logical_t == 40:
                SCENARIO_LOG.info(
                    "data_degraded window=missing_sensors logical_t=40..44 sensors=null for s2,s3"
                )
            if logical_t == 70:
                SCENARIO_LOG.info(
                    "data_degraded window=delayed_timestamps logical_t=70..79 timestamps lag 10 steps vs nominal"
                )
            if logical_t == 75:
                SCENARIO_LOG.info(
                    "data_degraded window=flatline logical_t>=75 sensor=s4 value=9.5 constant"
                )

            payload = {
                "timestamp": ts,
                "site_id": "pilot-scenario",
                "asset_id": "asset-1",
                "sensor_values": sensors,
            }

            result = service.ingest_payload(payload)
            action = str(result.get("action_state", ""))
            if first_watch is None and action == "WATCH":
                first_watch = logical_t
            if first_alert is None and action == "ALERT":
                first_alert = logical_t

            _log_engine_data_quality_if_degraded(
                logical_t=logical_t, ingest_seq=ingest_seq, result=result
            )
            _print_step_record(
                ingest_seq=ingest_seq,
                logical_t=logical_t,
                result=result,
                degraded=step_tags if step_tags else None,
                save_rows=save_rows,
            )
            ingest_seq += 1

            # Duplicated frame: same payload ingested twice (logical_t 55).
            if logical_t == 55:
                SCENARIO_LOG.info(
                    "data_degraded event=duplicated_frame logical_t=55 re-ingesting identical payload"
                )
                dup_payload = copy.deepcopy(payload)
                result_dup = service.ingest_payload(dup_payload)
                _log_engine_data_quality_if_degraded(
                    logical_t=logical_t, ingest_seq=ingest_seq, result=result_dup
                )
                _print_step_record(
                    ingest_seq=ingest_seq,
                    logical_t=logical_t,
                    result=result_dup,
                    degraded=["duplicated_frame"],
                    save_rows=save_rows,
                )
                ingest_seq += 1

        del service, store, engine
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _write_pilot_results(results_path, save_rows)

    # Progression summary (operator-facing heuristic status)
    print(
        json.dumps(
            {
                "summary": {
                    "scenario": "stable → regime_shift → instability",
                    "phases": {"stable_t_lt_30": True, "regime_shift_30_le_t_lt_60": True, "instability_t_ge_60": True},
                    "degradation_injected": {
                        "missing_s2_s3": "logical_t 40–44",
                        "duplicated_frame": "logical_t 55 (second ingest identical payload)",
                        "delayed_timestamps": "logical_t 70–79 (10-step lag)",
                        "s4_flatline": "logical_t ≥ 75",
                    },
                    "first_WATCH_step": first_watch,
                    "first_ALERT_step": first_alert,
                    "note": "status=action_state; state=engine drift alert state; "
                    "interpreted_state=decision layer; confidence=service operator confidence [0,1]. "
                    "Degradation logs on stderr (neraium_pilot.scenario).",
                }
            },
            indent=2,
        ),
        flush=True,
    )


def _run_file_payloads(path: Path, *, results_path: Path) -> None:
    """Original behavior: load JSON payload(s) and print pilot view."""
    payloads = _load_json_payloads(path)
    _setup_scenario_logging()

    tmp_dir = tempfile.mkdtemp(prefix="neraium_pilot_")
    save_rows: list[dict[str, Any]] = []
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
        for timestep, payload in enumerate(payloads):
            result = service.ingest_payload(payload)
            outputs.append(_pilot_view(result))
            save_rows.append(_save_result_row(timestep=timestep, result=result))

        del service, store, engine
        gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    _write_pilot_results(results_path, save_rows)

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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pilot_results.json"),
        help="Write per-step results (timestep, signals, state, interpreted_state, score) to this JSON file.",
    )
    args = parser.parse_args()

    os.environ["NERAIUM_PILOT_HARDENING"] = "1"
    if args.debug:
        os.environ["NERAIUM_DEBUG_PILOT"] = "1"

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(str(input_path))
        _run_file_payloads(input_path, results_path=args.output)
        return

    _run_scenario(
        timesteps=args.timesteps,
        seed=args.seed,
        baseline_window=args.baseline_window,
        recent_window=args.recent_window,
        results_path=args.output,
    )


if __name__ == "__main__":
    main()
