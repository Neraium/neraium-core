from __future__ import annotations

import csv
import math
import os
from csv import DictReader
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_TURBO_SOURCE = Path(
    "C:/Users/Owner/Downloads/WUR_AutonomousGreenhouseProject_EDA-main/"
    "WUR_AutonomousGreenhouseProject_EDA-main/iGrow/greenhouse_results_turbo.csv"
)
LOCAL_TURBO_SOURCE = REPO_ROOT / "greenhouse_demo" / "greenhouse_results_turbo.csv"
_REAL_REPLAY_CACHE: list[dict[str, Any]] | None = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _is_numeric_string(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return None
    if raw in {"true", "1", "yes", "admit", "admitted"}:
        return True
    if raw in {"false", "0", "no", "deny", "denied", "suppressed"}:
        return False
    return None


def _is_truthy(value: Any) -> bool:
    parsed = _coerce_optional_bool(value)
    return bool(parsed)


def _normalize_numeric_timestamp(value: float) -> str | None:
    if not math.isfinite(value):
        return None
    abs_value = abs(value)
    if abs_value > 10_000_000:
        epoch_seconds = value
        if abs_value >= 1_000_000_000_000_000_000:
            epoch_seconds = value / 1_000_000_000
        elif abs_value >= 1_000_000_000_000_000:
            epoch_seconds = value / 1_000_000
        elif abs_value >= 1_000_000_000_000:
            epoch_seconds = value / 1_000
        try:
            return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return (datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=value)).isoformat()
    except OverflowError:
        return None


def _normalize_timestamp(value: Any, index: int, base_time: datetime) -> str:
    try:
        raw = str(value or "").strip()
        if isinstance(value, (int, float)):
            normalized = _normalize_numeric_timestamp(float(value))
            if normalized:
                return normalized
        if raw and _is_numeric_string(raw):
            normalized = _normalize_numeric_timestamp(float(raw))
            if normalized:
                return normalized
        if raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, OverflowError):
        pass
    return (base_time + timedelta(minutes=index)).isoformat()


def _resolve_turbo_source() -> Path | None:
    env_source = os.getenv("GREENHOUSE_RESULTS_TURBO_CSV")
    candidates = [Path(env_source)] if env_source else []
    candidates.extend([WINDOWS_TURBO_SOURCE, LOCAL_TURBO_SOURCE])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_sensor_values_from_row(row: dict[str, Any]) -> dict[str, float]:
    sensor_values: dict[str, float] = {}
    non_sensor_cols = {
        "timestamp", "site_id", "asset_id", "state", "regime_name", "interpreted_state", "system_health", "risk_level",
        "signal_emitted", "event_admitted", "confidence", "confidence_score", "structural_drift_score", "relational_stability_score",
        "explanation", "explanation_text", "operator_message",
    }
    for col, value in row.items():
        if col in non_sensor_cols:
            continue
        if not value or not _is_numeric_string(str(value)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        sensor_values[col] = numeric
    return sensor_values


def _normalize_engine_result(result: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    drift = _clamp(_coerce_float(result.get("structural_drift_score"), _coerce_float(result.get("drift_score"), 0.0)))
    stability = _clamp(_coerce_float(result.get("relational_stability_score"), _coerce_float(result.get("stability_score"), 1.0)))
    confidence = _clamp(
        _coerce_float(
            result.get("confidence_score"),
            _coerce_float(result.get("confidence"), _clamp(0.62 + drift * 0.3)),
        )
    )
    admitted_override = _coerce_optional_bool(result.get("signal_emitted"))
    if admitted_override is None:
        admitted_override = _coerce_optional_bool(result.get("event_admitted"))
    admitted = admitted_override if admitted_override is not None else (drift >= 0.55 and stability <= 0.45)
    return {
        **result,
        "timestamp": frame["timestamp"],
        "site_id": frame["site_id"],
        "asset_id": frame["asset_id"],
        "sensor_values": frame["sensor_values"],
        "state": str(result.get("state") or _state_from_drift(drift)),
        "regime_name": str(result.get("regime_name") or result.get("interpreted_state") or result.get("state") or "greenhouse_demo"),
        "structural_drift_score": round(drift, 6),
        "relational_stability_score": round(stability, 6),
        "system_health": str(result.get("risk_level") or result.get("system_health") or _health_from_drift(drift)).lower(),
        "confidence_score": round(confidence, 6),
        "event_admitted": bool(admitted),
        "evidence_summary": str(
            result.get("explanation_text")
            or result.get("explanation")
            or result.get("operator_message")
            or "Processed telemetry replayed through StructuralEngine."
        ).strip(),
    }


def _run_structural_replay(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    engine = StructuralEngine()
    records: list[dict[str, Any]] = []
    for frame in frames:
        result = engine.process_frame(frame)
        if not isinstance(result, dict):
            continue
        records.append(_normalize_engine_result(result, frame))
    return records


def _load_rows_from_turbo_results(*, limit: int | None) -> list[dict[str, Any]]:
    source = _resolve_turbo_source()
    if source is None:
        return []
    csv.field_size_limit(10_000_000)
    with source.open("r", encoding="utf-8") as handle:
        raw_rows = list(DictReader(handle))

    if not raw_rows:
        return []
    base_time = datetime.now(timezone.utc) - timedelta(minutes=max(1, len(raw_rows)))
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        dynamic_signal_strength = _clamp(
            _coerce_float(
                row.get("dynamic_signal_strength"),
                _coerce_float(row.get("signal_strength"), _coerce_float(row.get("structural_drift_score"), 0.0)),
            )
        )
        confidence = _clamp(_coerce_float(row.get("confidence_score"), _coerce_float(row.get("confidence"), 0.0)))
        risk_level = str(row.get("risk_level") or "unknown").strip()
        phase = str(row.get("phase") or row.get("transition_state") or "unknown").strip()
        interpreted_state = str(row.get("interpreted_state") or row.get("state") or "unknown").strip()
        explanation_text = str(
            row.get("explanation_text") or row.get("operator_message") or row.get("explanation") or ""
        ).strip()
        if not explanation_text:
            explanation_text = "No explanation provided in greenhouse_results_turbo.csv."
        normalized.append(
            {
                "timestamp": _normalize_timestamp(row.get("timestamp"), idx, base_time),
                "site_id": str(row.get("site_id") or "grow-house"),
                "asset_id": str(row.get("asset_id") or "zone-A"),
                "system_phase": phase,
                "state": interpreted_state,
                "regime_name": phase or interpreted_state,
                "risk_level": risk_level,
                "system_health": risk_level.lower(),
                "confidence_score": confidence,
                "dynamic_signal_strength": dynamic_signal_strength,
                "structural_drift_score": dynamic_signal_strength,
                "relational_stability_score": round(1.0 - dynamic_signal_strength, 6),
                "event_admitted": _is_truthy(row.get("signal_emitted")),
                "transition_type": (phase or "STABLE").upper(),
                "evidence_summary": explanation_text,
                "explanation_text": explanation_text,
                "sensor_values": _parse_sensor_values_from_row(row),
            }
        )
    normalized.sort(key=lambda entry: str(entry.get("timestamp") or ""))
    if isinstance(limit, int) and limit > 0:
        return normalized[:limit]
    return normalized


def load_greenhouse_demo_bundle(*, limit: int | None = 320) -> tuple[list[dict[str, Any]], str]:
    global _REAL_REPLAY_CACHE
    if _REAL_REPLAY_CACHE is None:
        _REAL_REPLAY_CACHE = _load_rows_from_turbo_results(limit=None)
    rows = _REAL_REPLAY_CACHE or []
    source = _resolve_turbo_source()
    if isinstance(limit, int) and limit > 0:
        rows = rows[:limit]
    return rows, str(source) if source else "greenhouse_results_turbo.csv"


def load_greenhouse_demo_records(*, limit: int | None = 180) -> list[dict[str, Any]]:
    rows, _ = load_greenhouse_demo_bundle(limit=limit)
    return rows
