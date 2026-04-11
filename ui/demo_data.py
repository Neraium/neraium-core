from __future__ import annotations

import csv
import json
import runpy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from neraium_core.alignment import StructuralEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRAFAST_DEMO_SCRIPT = REPO_ROOT / "greenhouse_demo" / "run_grow_demo_ultrafast.py"
GREENHOUSE_SCENARIO_JSON = REPO_ROOT / "apps" / "api" / "demo_data" / "cannabis_grow_op_scenario.json"
IGROW_DIR = REPO_ROOT / "WUR_AutonomousGreenhouseProject_EDA-main" / "iGrow"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _normalize_sensor_values(sensor_values: Any) -> dict[str, float]:
    if not isinstance(sensor_values, dict):
        return {}
    return {str(key): float(value) for key, value in sensor_values.items() if isinstance(value, (int, float))}


def _health_from_drift(drift: float) -> str:
    if drift >= 0.8:
        return "critical"
    if drift >= 0.6:
        return "degraded"
    if drift >= 0.4:
        return "watch"
    return "nominal"


def _state_from_drift(drift: float) -> str:
    if drift >= 0.8:
        return "alert"
    if drift >= 0.6:
        return "instability"
    if drift >= 0.4:
        return "early_structural_drift"
    return "stable"


def _record_from_row(*, timestamp: str, site_id: str, asset_id: str, regime_name: str, sensor_values: dict[str, float]) -> dict[str, Any]:
    temperature = 24.0
    if "temperature" in sensor_values:
        temperature = float(sensor_values["temperature"])
    elif "temperature_f" in sensor_values:
        temperature = (float(sensor_values["temperature_f"]) - 32.0) * (5.0 / 9.0)

    humidity = float(sensor_values.get("humidity", sensor_values.get("humidity_rh", 55.0)))
    vapor = float(sensor_values.get("vapor_pressure_deficit", sensor_values.get("vpd_kpa", 1.0)))

    temp_risk = _clamp((temperature - 24.0) / 16.0)
    humidity_risk = _clamp(abs(humidity - 55.0) / 35.0)
    vapor_risk = _clamp(abs(vapor - 1.2) / 1.6)
    drift = round(_clamp((temp_risk * 0.55) + (humidity_risk * 0.25) + (vapor_risk * 0.20)), 6)
    stability = round(_clamp(1.0 - drift), 6)
    confidence = round(_clamp(0.62 + drift * 0.3), 6)

    return {
        "timestamp": timestamp,
        "site_id": site_id,
        "asset_id": asset_id,
        "state": _state_from_drift(drift),
        "regime_name": regime_name,
        "structural_drift_score": drift,
        "relational_stability_score": stability,
        "system_health": _health_from_drift(drift),
        "confidence_score": confidence,
        "event_admitted": bool(drift >= 0.55 and stability <= 0.45),
        "sensor_values": sensor_values,
    }


def _normalize_records(rows: Iterable[dict[str, Any]], *, limit: int | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized.append(
            _record_from_row(
                timestamp=str(row.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                site_id=str(row.get("site_id") or "grow-op-facility-01"),
                asset_id=str(row.get("asset_id") or "canopy-zone-A"),
                regime_name=str(row.get("regime_name") or row.get("state") or "greenhouse_demo"),
                sensor_values=_normalize_sensor_values(row.get("sensor_values")),
            )
        )
    normalized.sort(key=lambda entry: str(entry.get("timestamp") or ""))
    if isinstance(limit, int) and limit > 0:
        return normalized[:limit]
    return normalized


def _extract_rows_from_script_scope(scope: dict[str, Any], *, limit: int | None) -> list[dict[str, Any]]:
    candidates = (
        "build_ultrafast_demo_rows",
        "generate_ultrafast_demo_rows",
        "build_demo_rows",
        "generate_demo_rows",
        "load_demo_rows",
    )
    for name in candidates:
        fn = scope.get(name)
        if not callable(fn):
            continue
        code = getattr(fn, "__code__", None)
        accepts_limit = bool(code and "limit" in code.co_varnames)
        rows = fn(limit=limit) if accepts_limit else fn()
        if isinstance(rows, list) and rows:
            return _normalize_records(rows, limit=limit)
    return []


def _load_rows_from_ultrafast_script(*, limit: int | None) -> list[dict[str, Any]]:
    if not ULTRAFAST_DEMO_SCRIPT.is_file():
        return []
    try:
        scope = runpy.run_path(str(ULTRAFAST_DEMO_SCRIPT))
    except Exception:
        return []
    return _extract_rows_from_script_scope(scope, limit=limit)


def _load_rows_from_greenhouse_scenario(*, limit: int | None) -> list[dict[str, Any]]:
    if not GREENHOUSE_SCENARIO_JSON.exists():
        return []
    payload = json.loads(GREENHOUSE_SCENARIO_JSON.read_text(encoding="utf-8"))
    asset = payload.get("asset") or {}
    site_id = str(asset.get("site_id") or "grow-op-facility-01")
    asset_id = str(asset.get("asset_id") or "canopy-zone-A")

    scenario_rows: list[dict[str, Any]] = []
    for phase in payload.get("phases") or []:
        phase_name = str(phase.get("name") or "greenhouse_demo")
        for frame in phase.get("frames") or []:
            scenario_rows.append(
                {
                    "minute_offset": int(frame.get("minute_offset") or 0),
                    "regime_name": phase_name,
                    "sensor_values": _normalize_sensor_values(frame.get("sensor_values")),
                }
            )

    scenario_rows = [row for row in scenario_rows if row["sensor_values"]]
    scenario_rows.sort(key=lambda row: int(row["minute_offset"]))
    if isinstance(limit, int) and limit > 0:
        scenario_rows = scenario_rows[:limit]

    base_time = datetime.now(timezone.utc) - timedelta(minutes=max(1, len(scenario_rows)))
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(scenario_rows):
        rows.append(
            {
                "timestamp": (base_time + timedelta(minutes=idx)).isoformat(),
                "site_id": site_id,
                "asset_id": asset_id,
                "regime_name": row["regime_name"],
                "sensor_values": row["sensor_values"],
            }
        )
    return _normalize_records(rows, limit=limit)


def _parse_timestamp(value: str, fallback_index: int) -> float:
    raw = (value or "").strip()
    if not raw:
        return float(fallback_index)
    try:
        return float(raw)
    except ValueError:
        pass
    normalized = raw.replace("Z", "+00:00")
    for parser in (datetime.fromisoformat,):
        try:
            return parser(normalized).timestamp()
        except ValueError:
            continue
    return float(fallback_index)


def _is_numeric_string(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _discover_igrow_csv() -> Path | None:
    if not IGROW_DIR.is_dir():
        return None
    candidates = sorted(IGROW_DIR.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not candidates:
        return None
    priority = ["Greenhouse_climate", "climate", "sensor", "timeseries"]
    for key in priority:
        for candidate in candidates:
            if key.lower() in candidate.name.lower():
                return candidate
    return candidates[0]


def _load_igrow_frames(*, limit: int | None) -> tuple[list[dict[str, Any]], Path | None]:
    csv_path = _discover_igrow_csv()
    if csv_path is None:
        return [], None

    frames: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], csv_path

        fieldnames = [str(name) for name in reader.fieldnames]
        ts_col = next((c for c in fieldnames if "time" in c.lower() or "date" in c.lower()), fieldnames[0])

        for idx, row in enumerate(reader):
            sensor_values: dict[str, float] = {}
            for col, raw in row.items():
                if col is None or col == ts_col:
                    continue
                value = (raw or "").strip()
                if not value or not _is_numeric_string(value):
                    continue
                sensor_values[col] = float(value)
            if not sensor_values:
                continue

            timestamp = _parse_timestamp(str(row.get(ts_col) or ""), idx)
            frames.append(
                {
                    "timestamp": timestamp,
                    "site_id": "greenhouse",
                    "asset_id": "igrow",
                    "sensor_values": sensor_values,
                }
            )
            if isinstance(limit, int) and limit > 0 and len(frames) >= limit:
                break

    return frames, csv_path


def _curated_slice(records: list[dict[str, Any]], *, target: int = 240) -> list[dict[str, Any]]:
    if len(records) <= target:
        return records

    quarter = max(10, target // 4)
    stable = [r for r in records if float(r.get("structural_drift_score") or 0.0) < 0.25][:quarter]
    transition = [r for r in records if 0.25 <= float(r.get("structural_drift_score") or 0.0) < 0.5][:quarter]
    divergence = [r for r in records if 0.5 <= float(r.get("structural_drift_score") or 0.0) < 0.75][:quarter]
    reorg = [r for r in records if float(r.get("structural_drift_score") or 0.0) >= 0.75][:quarter]

    curated = stable + transition + divergence + reorg
    if len(curated) < target:
        remaining = [r for r in records if r not in curated]
        curated.extend(remaining[: target - len(curated)])
    return curated[:target]


def _run_structural_replay(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not frames:
        return []

    engine = StructuralEngine(baseline_window=20, recent_window=6)
    records: list[dict[str, Any]] = []
    for frame in frames:
        result = engine.process_frame(frame)
        if not isinstance(result, dict):
            continue
        result["timestamp"] = frame["timestamp"]
        result["site_id"] = frame["site_id"]
        result["asset_id"] = frame["asset_id"]
        result["sensor_values"] = frame["sensor_values"]
        records.append(result)
    return records


def load_greenhouse_demo_bundle(*, limit: int | None = 320, curated: bool = True) -> tuple[list[dict[str, Any]], str]:
    igrow_frames, csv_path = _load_igrow_frames(limit=limit)
    if igrow_frames:
        records = _run_structural_replay(igrow_frames)
        if curated:
            records = _curated_slice(records)
        if records:
            source_label = str(csv_path.relative_to(REPO_ROOT)) if csv_path is not None else "iGrow/*.csv"
            return records, source_label

    ultrafast = _load_rows_from_ultrafast_script(limit=limit)
    if ultrafast:
        return ultrafast, str(ULTRAFAST_DEMO_SCRIPT.relative_to(REPO_ROOT))

    scenario = _load_rows_from_greenhouse_scenario(limit=limit)
    return scenario, str(GREENHOUSE_SCENARIO_JSON.relative_to(REPO_ROOT))


def load_greenhouse_demo_records(*, limit: int | None = 320, curated: bool = True) -> list[dict[str, Any]]:
    rows, _ = load_greenhouse_demo_bundle(limit=limit, curated=curated)
    return rows
