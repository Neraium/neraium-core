import csv
import math
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional

DEFAULT_SITE_ID = "default-site"
DEFAULT_ASSET_ID = "default-asset"
REQUIRED_CSV_COLUMNS = {"timestamp", "site_id", "asset_id"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return now_iso()

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp: {value!r}") from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def normalize_identifier(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)
    return safe or default


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        value = text

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(parsed):
        return None
    return parsed


def build_frame(
    timestamp: Any,
    site_id: Any,
    asset_id: Any,
    sensor_values: Dict[Any, Any],
    include_sensor_quality: bool = True,
) -> Dict[str, Any]:
    if not isinstance(sensor_values, dict):
        raise ValueError("sensor_values must be an object")

    frame: Dict[str, Any] = {
        "timestamp": normalize_timestamp(timestamp),
        "site_id": normalize_identifier(site_id, DEFAULT_SITE_ID),
        "asset_id": normalize_identifier(asset_id, DEFAULT_ASSET_ID),
        "sensor_values": {},
    }
    if include_sensor_quality:
        frame["sensor_quality"] = {}

    for key, value in sensor_values.items():
        sensor_name = str(key).strip()
        if not sensor_name:
            raise ValueError("Sensor name cannot be empty")
        numeric_value = coerce_float(value)
        frame["sensor_values"][sensor_name] = numeric_value
        if include_sensor_quality:
            frame["sensor_quality"][sensor_name] = "ok" if numeric_value is not None else "missing"

    return frame


def normalize_rest_payload(payload: Dict[str, Any], include_sensor_quality: bool = True) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")

    return build_frame(
        timestamp=payload.get("timestamp"),
        site_id=payload.get("site_id", DEFAULT_SITE_ID),
        asset_id=payload.get("asset_id", DEFAULT_ASSET_ID),
        sensor_values=payload.get("sensor_values", {}),
        include_sensor_quality=include_sensor_quality,
    )


def parse_csv_text(csv_text: str, include_sensor_quality: bool = True) -> List[Dict[str, Any]]:
    if not isinstance(csv_text, str):
        raise ValueError("csv_text must be a string")

    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        return []

    headers = {h.strip() for h in reader.fieldnames if h is not None}
    if not REQUIRED_CSV_COLUMNS.issubset(headers):
        missing = sorted(REQUIRED_CSV_COLUMNS - headers)
        raise ValueError(f"CSV missing required columns: {missing}")

    sensor_columns = [
        col for col in reader.fieldnames if col is not None and col.strip() not in REQUIRED_CSV_COLUMNS
    ]
    frames: List[Dict[str, Any]] = []

    for row_index, row in enumerate(reader, start=2):
        if row is None:
            continue

        sensor_values: Dict[str, Any] = {col.strip(): row.get(col) for col in sensor_columns}
        try:
            frame = build_frame(
                timestamp=row.get("timestamp"),
                site_id=row.get("site_id"),
                asset_id=row.get("asset_id"),
                sensor_values=sensor_values,
                include_sensor_quality=include_sensor_quality,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid CSV row {row_index}: {exc}") from exc

        frames.append(frame)

    return frames
