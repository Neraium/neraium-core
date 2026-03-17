from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List

DEFAULT_SITE_ID = "default-site"
DEFAULT_ASSET_ID = "default-asset"
REQUIRED_CSV_COLUMNS = {"timestamp", "site_id", "asset_id"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_timestamp(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return now_iso()

    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_identifier(value: Any, default: str) -> str:
    if value is None:
        return default
    normalized = " ".join(str(value).split()).strip()
    return normalized or default


def normalize_sensor_name(value: Any) -> str:
    name = str(value).strip()
    if not name:
        raise ValueError("Sensor name cannot be empty")
    return name


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_frame(
    *,
    timestamp: Any,
    site_id: Any,
    asset_id: Any,
    sensor_values: Dict[Any, Any],
    include_sensor_quality: bool = False,
) -> Dict[str, Any]:
    if not isinstance(sensor_values, dict):
        raise ValueError("sensor_values must be an object")

    frame: Dict[str, Any] = {
        "timestamp": normalize_timestamp(timestamp),
        "site_id": normalize_identifier(site_id, DEFAULT_SITE_ID),
        "asset_id": normalize_identifier(asset_id, DEFAULT_ASSET_ID),
        "sensor_values": {},
    }

    sensor_quality: Dict[str, str] = {}
    for raw_key, raw_value in sensor_values.items():
        name = normalize_sensor_name(raw_key)
        numeric = coerce_float(raw_value)
        frame["sensor_values"][name] = numeric
        sensor_quality[name] = "ok" if numeric is not None else "missing"

    if include_sensor_quality:
        frame["sensor_quality"] = sensor_quality

    return frame


def normalize_rest_payload(
    payload: Dict[str, Any],
    include_sensor_quality: bool = False,
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")

    return build_frame(
        timestamp=payload.get("timestamp"),
        site_id=payload.get("site_id", DEFAULT_SITE_ID),
        asset_id=payload.get("asset_id", DEFAULT_ASSET_ID),
        sensor_values=payload.get("sensor_values", {}),
        include_sensor_quality=include_sensor_quality,
    )


def parse_csv_text(csv_text: str, include_sensor_quality: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(csv_text, str):
        raise ValueError("csv_text must be a string")

    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        return []

    headers = {h.strip() for h in reader.fieldnames if h}
    missing_required = sorted(REQUIRED_CSV_COLUMNS - headers)
    if missing_required:
        raise ValueError(
            "CSV must include timestamp, site_id, asset_id columns. "
            f"Missing: {missing_required}"
        )

    sensor_columns = [h for h in reader.fieldnames if h and h.strip() not in REQUIRED_CSV_COLUMNS]

    frames: List[Dict[str, Any]] = []
    row_errors: List[str] = []

    for row_num, row in enumerate(reader, start=2):
        sensor_values: Dict[str, Any] = {}
        for column in sensor_columns:
            sensor_values[column.strip()] = row.get(column)

        try:
            frames.append(
                build_frame(
                    timestamp=row.get("timestamp"),
                    site_id=row.get("site_id"),
                    asset_id=row.get("asset_id"),
                    sensor_values=sensor_values,
                    include_sensor_quality=include_sensor_quality,
                )
            )
        except ValueError as exc:
            row_errors.append(f"row {row_num}: {exc}")

    if row_errors:
        raise ValueError("CSV validation failed: " + "; ".join(row_errors))

    return frames
