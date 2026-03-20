import csv
import math
import os
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional


DEFAULT_SITE_ID = "default-site"
DEFAULT_ASSET_ID = "default-asset"
REQUIRED_CSV_COLUMNS = {"timestamp", "site_id", "asset_id"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(value: Any) -> str:
    """
    Normalize a timestamp into an ISO-8601 UTC string.
    Accepts datetime objects or strings. Falls back to current UTC time
    only when the input is None or empty.
    """
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
    return text if text else default


def normalize_sensor_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Sensor name cannot be empty")
    return text


def pilot_hardening_enabled() -> bool:
    """
    Pilot hardening feature toggle.

    When enabled, the pipeline rejects non-numeric sensor values and treats NaN/inf
    as missing (`None`) to keep downstream analytics stable.
    """

    v = os.getenv("NERAIUM_PILOT_HARDENING", "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def coerce_float(value: Any, *, sensor_name: str) -> Optional[float]:
    """
    Convert a sensor input value into a float.

    Returns:
      - `None` for missing values (`None`, empty string).
      - In pilot mode, rejects malformed non-numeric values with `ValueError`.
      - In pilot mode, converts NaN/inf to `None`.
    """

    strict = pilot_hardening_enabled()

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            f = float(text)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Invalid signal value for {sensor_name!r}: {value!r}") from exc
            return None
    elif isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Invalid signal value for {sensor_name!r}: {value!r}") from exc
            return None
    else:
        if strict:
            raise ValueError(f"Invalid signal type for {sensor_name!r}: {type(value).__name__}")
        return None

    if strict and (math.isnan(f) or math.isinf(f)):
        return None

    return f


def build_frame(
    timestamp: Any,
    site_id: Any,
    asset_id: Any,
    sensor_values: Dict[Any, Any],
    
    
) -> Dict[str, Any]:
    """
    Build the internal telemetry frame for `StructuralEngine.process_frame()`.

    Internal contract:
    - `frame["timestamp"]` is an ISO-8601 UTC string
    - `frame["sensor_values"]` is a dict of `{signal_name: float | None}`
    """
    if not isinstance(sensor_values, dict):
        raise ValueError("sensor_values must be an object")

    # Internal frame shape used by `StructuralEngine.process_frame`.
    # Keep this stable across pipelines/entrypoints so production ingestion works.
    frame: Dict[str, Any] = {
        "timestamp": normalize_timestamp(timestamp),
        "site_id": site_id,
        "asset_id": asset_id,
        "sensor_values": {},
        "sensor_quality": {},
        "aligned": [],
        "anomaly": False,
    }

    for raw_key, raw_value in sensor_values.items():
        sensor_name = normalize_sensor_name(raw_key)
        numeric_value = coerce_float(raw_value, sensor_name=sensor_name)

        frame["sensor_values"][sensor_name] = numeric_value
        frame["sensor_quality"][sensor_name] = "ok" if numeric_value is not None else "missing"

    return frame


def normalize_rest_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an incoming REST payload into the internal frame format.

    In pilot hardening mode (`NERAIUM_PILOT_HARDENING=1`), validation is strict:
    - `sensor_values` must be an object/dict
    - sensor values must be numeric or numeric strings (or `null`)
    - invalid values are rejected with clear `ValueError` messages
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")

    return build_frame(
        timestamp=payload.get("timestamp"),
        site_id=payload.get("site_id", DEFAULT_SITE_ID),
        asset_id=payload.get("asset_id", DEFAULT_ASSET_ID),
        sensor_values=payload.get("sensor_values", {}),
    )


def parse_csv_text(csv_text: str) -> List[Dict[str, Any]]:
    """
    Parse CSV text into a list of normalized internal frames.

    Required columns:
        timestamp, site_id, asset_id

    All remaining columns are treated as sensor columns.
    """
    if not isinstance(csv_text, str):
        raise ValueError("csv_text must be a string")

    reader = csv.DictReader(StringIO(csv_text))

    if reader.fieldnames is None:
        return []

    headers = {h.strip() for h in reader.fieldnames if h is not None}

    if not REQUIRED_CSV_COLUMNS.issubset(headers):
        missing = sorted(REQUIRED_CSV_COLUMNS - headers)
        raise ValueError(
            f"CSV must include timestamp, site_id, asset_id columns. Missing: {missing}"
        )

    sensor_columns = [
        h for h in reader.fieldnames
        if h is not None and h.strip() not in REQUIRED_CSV_COLUMNS
    ]

    frames: List[Dict[str, Any]] = []

    for row_index, row in enumerate(reader, start=2):
        if row is None:
            continue

        sensor_values: Dict[str, Any] = {}
        for col in sensor_columns:
            sensor_values[col.strip()] = row.get(col)

        try:
            frame = build_frame(
                timestamp=row.get("timestamp"),
                site_id=row.get("site_id"),
                asset_id=row.get("asset_id"),
                sensor_values=sensor_values,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid CSV row {row_index}: {exc}") from exc

        frames.append(frame)

    return frames