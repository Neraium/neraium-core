from __future__ import annotations

import csv
import json
import runpy
import csv
from csv import DictReader
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from neraium_core.alignment import StructuralEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRAFAST_DEMO_SCRIPT = REPO_ROOT / "greenhouse_demo" / "run_grow_demo_ultrafast.py"
GREENHOUSE_SCENARIO_JSON = REPO_ROOT / "apps" / "api" / "demo_data" / "cannabis_grow_op_scenario.json"
WUR_TURBO_SOURCE = REPO_ROOT / "WUR_AutonomousGreenhouseProject_EDA-main" / "iGrow" / "greenhouse_results_turbo"
LOCAL_TURBO_SOURCE = REPO_ROOT / "greenhouse_demo" / "greenhouse_results_turbo.csv"


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


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_timestamp(value: Any, index: int, base_time: datetime) -> str:
    try:
        if isinstance(value, (int, float)):
            # greenhouse_results_turbo uses Excel-style serial day values.
            return (datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=float(value))).isoformat()
        raw = str(value or "").strip()
        if raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass
    return (base_time + timedelta(minutes=index)).isoformat()


def _resolve_turbo_source() -> Path | None:
    candidates = [
        WUR_TURBO_SOURCE,
        WUR_TURBO_SOURCE.with_suffix(".csv"),
        WUR_TURBO_SOURCE / "greenhouse_results_turbo.csv",
        LOCAL_TURBO_SOURCE,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_rows_from_turbo_results(*, limit: int | None) -> list[dict[str, Any]]:
    source = _resolve_turbo_source()
    if source is None:
        return []
    csv.field_size_limit(10_000_000)
    with source.open("r", encoding="utf-8") as handle:
        raw_rows = list(DictReader(handle))

    if not raw_rows:
        return []
    if isinstance(limit, int) and limit > 0:
        raw_rows = raw_rows[:limit]

    base_time = datetime.now(timezone.utc) - timedelta(minutes=max(1, len(raw_rows)))
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        drift = _clamp(_coerce_float(row.get("structural_drift_score")))
        stability = _clamp(_coerce_float(row.get("relational_stability_score"), 1.0))
        confidence = _clamp(_coerce_float(row.get("confidence_score"), _coerce_float(row.get("confidence"), 0.6)))
        regime_name = str(row.get("regime_name") or row.get("interpreted_state") or row.get("state") or "greenhouse_demo")
        evidence = str(row.get("explanation_text") or row.get("explanation") or row.get("operator_message") or "").strip()

        admitted_raw = str(row.get("signal_emitted") or row.get("event_admitted") or "").strip().lower()
        admitted = admitted_raw in {"true", "1", "yes", "admit", "admitted"} or (drift >= 0.55 and stability <= 0.45)
        if admitted and not evidence:
            evidence = "Processed transition evidence exceeded gate threshold."
        if not evidence:
            evidence = "Processed telemetry ingested from greenhouse_results_turbo."

        normalized.append(
            {
                "timestamp": _normalize_timestamp(row.get("timestamp"), idx, base_time),
                "site_id": str(row.get("site_id") or "grow-house"),
                "asset_id": str(row.get("asset_id") or "zone-A"),
                "state": str(row.get("state") or _state_from_drift(drift)),
                "regime_name": regime_name,
                "structural_drift_score": round(drift, 6),
                "relational_stability_score": round(stability, 6),
                "system_health": str(row.get("risk_level") or row.get("system_health") or _health_from_drift(drift)).lower(),
                "confidence_score": round(confidence, 6),
                "event_admitted": bool(admitted),
                "evidence_summary": evidence,
                "sensor_values": {},
            }
        )
    normalized.sort(key=lambda entry: str(entry.get("timestamp") or ""))
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


def load_greenhouse_demo_bundle(*, limit: int | None = 180, curated: bool = True) -> tuple[list[dict[str, Any]], str]:
    """Load UI-ready greenhouse replay rows and the source label used to produce them."""
    del curated  # Compatibility shim; curated slicing is no longer applied for turbo/script/scenario data.

    turbo = _load_rows_from_turbo_results(limit=limit)
    if turbo:
        source = _resolve_turbo_source()
        source_label = str(source.relative_to(REPO_ROOT)) if source is not None else "greenhouse_results_turbo.csv"
        return turbo, source_label

    ultrafast = _load_rows_from_ultrafast_script(limit=limit)
    if ultrafast:
        return ultrafast, str(ULTRAFAST_DEMO_SCRIPT.relative_to(REPO_ROOT))

    scenario = _load_rows_from_greenhouse_scenario(limit=limit)
    return scenario, str(GREENHOUSE_SCENARIO_JSON.relative_to(REPO_ROOT))


def load_greenhouse_demo_records(*, limit: int | None = 180, curated: bool = True) -> list[dict[str, Any]]:
    rows, _ = load_greenhouse_demo_bundle(limit=limit, curated=curated)
    return rows
