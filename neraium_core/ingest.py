from __future__ import annotations

import csv
import re
from datetime import datetime, timezone
from io import StringIO
from typing import Any

from neraium_core.models import TelemetryFrame

SENSOR_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_\-:.]+")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_timestamp(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return now_iso()
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp '{value}': expected ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_identifier(value: Any, default: str) -> str:
    candidate = default if value is None else str(value).strip()
    if not candidate:
        candidate = default
    sanitized = IDENTIFIER_RE.sub("-", candidate)
    return sanitized[:128]


def normalize_sensor_name(value: Any) -> str:
    if value is None:
        raise ValueError("sensor name cannot be null")
    name = str(value).strip()
    if not SENSOR_NAME_RE.match(name):
        raise ValueError(f"invalid sensor name '{value}'")
    return name


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_frame(
    timestamp: Any,
    site_id: Any,
    asset_id: Any,
    sensor_values: dict[str, Any],
    *,
    include_quality: bool = True,
) -> TelemetryFrame:
    normalized_values: dict[str, float | None] = {}
    quality: dict[str, str] = {}

    for raw_name, raw_value in sensor_values.items():
        name = normalize_sensor_name(raw_name)
        number = coerce_float(raw_value)
        normalized_values[name] = number
        if include_quality:
            quality[name] = "ok" if number is not None else "missing"

    return TelemetryFrame(
        timestamp=normalize_timestamp(timestamp),
        site_id=normalize_identifier(site_id, "default-site"),
        asset_id=normalize_identifier(asset_id, "default-asset"),
        sensor_values=normalized_values,
        sensor_quality=quality,
    )


def normalize_rest_payload(payload: dict[str, Any]) -> TelemetryFrame:
    sensor_values = payload.get("sensor_values", {})
    if not isinstance(sensor_values, dict):
        raise ValueError("sensor_values must be an object")
    return build_frame(
        payload.get("timestamp"),
        payload.get("site_id"),
        payload.get("asset_id"),
        sensor_values,
    )


def parse_csv_text(csv_text: str) -> list[TelemetryFrame]:
    reader = csv.DictReader(StringIO(csv_text))
    if not reader.fieldnames:
        return []

    required = {"timestamp", "site_id", "asset_id"}
    missing = required - set(reader.fieldnames)
    if missing:
        missing_headers = ", ".join(sorted(missing))
        raise ValueError(f"CSV missing required columns: {missing_headers}")

    sensor_columns = [col for col in reader.fieldnames if col not in required]
    frames: list[TelemetryFrame] = []

    for row_idx, row in enumerate(reader, start=2):
        if row is None:
            continue

        sensor_values: dict[str, Any] = {}
        for col in sensor_columns:
            try:
                sensor_name = normalize_sensor_name(col)
            except ValueError as exc:
                raise ValueError(f"row {row_idx}: {exc}") from exc
            sensor_values[sensor_name] = row.get(col)

        try:
            frame = build_frame(
                row.get("timestamp"),
                row.get("site_id"),
                row.get("asset_id"),
                sensor_values,
            )
        except ValueError as exc:
            raise ValueError(f"row {row_idx}: {exc}") from exc
        frames.append(frame)

    return frames
