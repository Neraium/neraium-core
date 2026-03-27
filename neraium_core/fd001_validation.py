from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from neraium_core.sii import SIIEngine

EXPECTED_FD001_COLUMNS = 26
OPERATING_SETTING_COUNT = 3
SENSOR_COUNT = 21


@dataclass(frozen=True)
class Fd001Row:
    unit_id: int
    cycle: int
    operating_settings: tuple[float, float, float]
    sensors: tuple[float, ...]


def parse_fd001_line(line: str) -> Fd001Row:
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty line")
    if len(parts) < EXPECTED_FD001_COLUMNS:
        raise ValueError(
            f"FD001 line must contain at least {EXPECTED_FD001_COLUMNS} columns, got {len(parts)}"
        )

    values = [float(x) for x in parts[:EXPECTED_FD001_COLUMNS]]
    unit_id = int(values[0])
    cycle = int(values[1])
    operating_settings = tuple(values[2:5])
    sensors = tuple(values[5 : 5 + SENSOR_COUNT])
    return Fd001Row(unit_id=unit_id, cycle=cycle, operating_settings=operating_settings, sensors=sensors)


def load_fd001_dataset(path: str | Path) -> list[Fd001Row]:
    rows: list[Fd001Row] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if raw_line.strip() == "":
            continue
        rows.append(parse_fd001_line(raw_line))
    return rows


def group_rows_by_unit(rows: Iterable[Fd001Row]) -> dict[int, list[Fd001Row]]:
    grouped: dict[int, list[Fd001Row]] = {}
    for row in rows:
        grouped.setdefault(row.unit_id, []).append(row)
    for unit_id in list(grouped):
        grouped[unit_id] = sorted(grouped[unit_id], key=lambda r: r.cycle)
    return grouped


def fd001_row_to_payload(
    row: Fd001Row,
    *,
    site_id: str = "cmapss-fd001",
    start_time: datetime | None = None,
) -> dict[str, Any]:
    base_time = start_time or datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamp = base_time + timedelta(minutes=row.cycle)

    # Explicit deterministic mapping from CMAPSS FD001 columns to canonical Neraium payload keys.
    sensor_values: dict[str, float] = {
        "setting_1": float(row.operating_settings[0]),
        "setting_2": float(row.operating_settings[1]),
        "setting_3": float(row.operating_settings[2]),
    }
    for idx, value in enumerate(row.sensors, start=1):
        sensor_values[f"s{idx}"] = float(value)

    return {
        "timestamp": timestamp.isoformat(),
        "site_id": site_id,
        "asset_id": f"unit_{row.unit_id:03d}",
        "sensor_values": sensor_values,
    }


def _top_hypothesis(causal_analysis: dict[str, Any]) -> tuple[Any, Any]:
    if not isinstance(causal_analysis, dict):
        return None, None
    top = causal_analysis.get("top_hypothesis")
    if isinstance(top, dict):
        return top.get("id") or top.get("hypothesis"), top.get("confidence")
    hypotheses = causal_analysis.get("hypotheses") or causal_analysis.get("top_hypotheses")
    if isinstance(hypotheses, list) and hypotheses:
        candidate = hypotheses[0]
        if isinstance(candidate, dict):
            return candidate.get("id") or candidate.get("hypothesis"), candidate.get("confidence")
    return None, None


def _top_attribution_driver(attribution: dict[str, Any]) -> Any:
    if not isinstance(attribution, dict):
        return None
    top_sensors = attribution.get("top_sensors")
    if isinstance(top_sensors, list) and top_sensors:
        first = top_sensors[0]
        if isinstance(first, dict):
            return first.get("sensor")
        return first
    top_drivers = attribution.get("top_drivers")
    if isinstance(top_drivers, list) and top_drivers:
        first = top_drivers[0]
        if isinstance(first, dict):
            return first.get("sensor") or first.get("driver")
        return first
    return None


def flatten_validation_result(result: dict[str, Any], *, unit_id: int, cycle: int) -> dict[str, Any]:
    decision = result.get("decision") if isinstance(result.get("decision"), dict) else {}
    risk = result.get("risk_assessment") if isinstance(result.get("risk_assessment"), dict) else {}
    causal = result.get("causal_analysis") if isinstance(result.get("causal_analysis"), dict) else {}
    attribution = result.get("attribution") if isinstance(result.get("attribution"), dict) else {}

    hypothesis_id, hypothesis_confidence = _top_hypothesis(causal)
    return {
        "unit_id": unit_id,
        "cycle": cycle,
        "decision_action": decision.get("action"),
        "decision_confidence": decision.get("confidence"),
        "risk_current_level": risk.get("current_risk_level") or risk.get("risk_level"),
        "risk_trend": risk.get("projected_near_term_trend") or risk.get("trend"),
        "top_hypothesis_id": hypothesis_id,
        "top_hypothesis_confidence": hypothesis_confidence,
        "top_attribution_driver": _top_attribution_driver(attribution),
    }


def replay_fd001_units(
    grouped_rows: dict[int, list[Fd001Row]],
    *,
    unit_ids: list[int] | None = None,
    max_cycles: int | None = None,
    site_id: str = "cmapss-fd001",
    start_time: datetime | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    replay_units = sorted(unit_ids) if unit_ids else sorted(grouped_rows)
    for unit_id in replay_units:
        rows = grouped_rows.get(unit_id, [])
        if max_cycles is not None and max_cycles > 0:
            rows = rows[: int(max_cycles)]

        # Per-unit engine reset preserves sequential behavior and avoids cross-unit leakage.
        engine = SIIEngine()
        for row in rows:
            payload = fd001_row_to_payload(row, site_id=site_id, start_time=start_time)
            out = engine.process_payload(payload)
            full = {
                "unit_id": unit_id,
                "cycle": row.cycle,
                "attribution": out.get("attribution"),
                "regime_memory": out.get("regime_memory"),
                "risk_assessment": out.get("risk_assessment"),
                "operator_guidance": out.get("operator_guidance"),
                "causal_analysis": out.get("causal_analysis"),
                "decision": out.get("decision"),
            }
            full_results.append(full)
            summary_rows.append(flatten_validation_result(out, unit_id=unit_id, cycle=row.cycle))
        engine.close()

    return full_results, summary_rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "unit_id",
        "cycle",
        "decision_action",
        "decision_confidence",
        "risk_current_level",
        "risk_trend",
        "top_hypothesis_id",
        "top_hypothesis_confidence",
        "top_attribution_driver",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})
