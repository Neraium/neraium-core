import csv
import math
import os
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Optional


DEFAULT_SITE_ID = "default-site"
DEFAULT_ASSET_ID = "default-asset"
DEFAULT_CUSTOMER_ID = "default-customer"
# Legacy: ingest no longer requires these literal header names — use semantic mapping in csv_mapping.py.
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
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = None
        if dt is None:
            # Unix epoch seconds or milliseconds (common in exports)
            try:
                num = float(text)
                if 1e9 <= abs(num) <= 1e12:
                    dt = datetime.fromtimestamp(num, tz=timezone.utc)
                elif 1e12 < abs(num) <= 1e15:
                    dt = datetime.fromtimestamp(num / 1000.0, tz=timezone.utc)
            except (ValueError, OSError, OverflowError):
                dt = None
        if dt is None:
            raise ValueError(f"Invalid timestamp: {value!r}")

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
    customer_id: Any = DEFAULT_CUSTOMER_ID,
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
        "customer_id": normalize_identifier(customer_id, DEFAULT_CUSTOMER_ID),
        "site_id": normalize_identifier(site_id, DEFAULT_SITE_ID),
        "asset_id": normalize_identifier(asset_id, DEFAULT_ASSET_ID),
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
        customer_id=payload.get("customer_id", DEFAULT_CUSTOMER_ID),
        site_id=payload.get("site_id", DEFAULT_SITE_ID),
        asset_id=payload.get("asset_id", DEFAULT_ASSET_ID),
        sensor_values=payload.get("sensor_values", {}),
    )


def parse_csv_text(
    csv_text: str,
    *,
    customer_id: str | None = None,
    column_mapping: dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Parse CSV text into a list of normalized internal frames.

    Column roles are resolved via semantic mapping (see :mod:`neraium_core.csv_mapping`):
    one timestamp column, one asset/entity column, optional site, and one or more sensor columns.

    If ``column_mapping`` is omitted, roles are inferred from headers (and sample values when needed).
    """
    if not isinstance(csv_text, str):
        raise ValueError("csv_text must be a string")

    from neraium_core.csv_mapping import resolve_mapping, row_to_frame_kwargs

    reader = csv.DictReader(StringIO(csv_text))

    if reader.fieldnames is None:
        return []

    headers = [h for h in reader.fieldnames if h is not None]
    resolved_customer = customer_id or DEFAULT_CUSTOMER_ID
    sample_snippet = csv_text[:65536] if column_mapping is None else None
    try:
        mapping, _warnings = resolve_mapping(headers, column_mapping, csv_sample=sample_snippet)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    frames: List[Dict[str, Any]] = []

    for row_index, row in enumerate(reader, start=2):
        if row is None:
            continue

        try:
            kwargs = row_to_frame_kwargs(row, mapping, customer_id=resolved_customer)
            frame = build_frame(**kwargs)
        except ValueError as exc:
            raise ValueError(f"Invalid CSV row {row_index}: {exc}") from exc

        frames.append(frame)

    return frames