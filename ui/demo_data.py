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


def _normalize_timestamp(value: Any, index: int, base_time: datetime) -> str:
    try:
        raw = str(value or "").strip()
        if isinstance(value, (int, float)):
            return (datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=float(value))).isoformat()
        if raw and _is_numeric_string(raw):
            serial = float(raw)
            if math.isfinite(serial):
                return (datetime(1899, 12, 30, tzinfo=timezone.utc) + timedelta(days=serial)).isoformat()
        if raw:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
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


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes", "admit", "admitted"}


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
