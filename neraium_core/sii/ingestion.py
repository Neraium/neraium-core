from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from .errors import SIIValidationError
from .types import TelemetryFrame


REQUIRED_CSV_COLUMNS = {"timestamp", "site_id", "asset_id"}


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            fv = float(text)
            if not (fv == fv) or abs(fv) == float("inf"):
                return None
            return fv
        except ValueError as exc:
            raise SIIValidationError(f"Invalid numeric value: {value!r}") from exc
    if isinstance(value, (int, float)):
        fv = float(value)
        if not (fv == fv) or abs(fv) == float("inf"):
            return None
        return fv
    raise SIIValidationError(f"Unsupported value type: {type(value).__name__}")


def frame_from_payload(payload: dict[str, Any]) -> TelemetryFrame:
    if not isinstance(payload, dict):
        raise SIIValidationError("Payload must be an object")

    sensor_values = payload.get("sensor_values")
    if not isinstance(sensor_values, dict):
        raise SIIValidationError("sensor_values must be an object")

    parsed_sensors: dict[str, float | None] = {}
    for raw_name, raw_value in sensor_values.items():
        name = str(raw_name).strip()
        if not name:
            raise SIIValidationError("Sensor name cannot be empty")
        parsed_sensors[name] = _as_float_or_none(raw_value)

    return TelemetryFrame(
        timestamp=str(payload.get("timestamp", "")).strip() or "0",
        site_id=str(payload.get("site_id", "default-site")).strip() or "default-site",
        asset_id=str(payload.get("asset_id", "default-asset")).strip() or "default-asset",
        sensor_values=parsed_sensors,
    )


def frames_from_csv(csv_text: str) -> list[TelemetryFrame]:
    if not isinstance(csv_text, str):
        raise SIIValidationError("csv_text must be a string")

    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        return []

    headers = {h.strip() for h in reader.fieldnames if h is not None}
    if not REQUIRED_CSV_COLUMNS.issubset(headers):
        missing = sorted(REQUIRED_CSV_COLUMNS - headers)
        raise SIIValidationError(f"CSV missing required columns: {missing}")

    header_lookup: dict[str, str] = {}
    for h in reader.fieldnames:
        if h is None:
            continue
        norm = h.strip()
        if norm and norm not in header_lookup:
            header_lookup[norm] = h

    sensor_columns = [h for h in header_lookup.keys() if h not in REQUIRED_CSV_COLUMNS]

    out: list[TelemetryFrame] = []
    for row_index, row in enumerate(reader, start=2):
        sensor_values = {col: row.get(header_lookup[col]) for col in sensor_columns}
        payload = {
            "timestamp": row.get(header_lookup.get("timestamp", "timestamp")),
            "site_id": row.get(header_lookup.get("site_id", "site_id")),
            "asset_id": row.get(header_lookup.get("asset_id", "asset_id")),
            "sensor_values": sensor_values,
        }
        try:
            out.append(frame_from_payload(payload))
        except SIIValidationError as exc:
            raise SIIValidationError(f"Invalid CSV row {row_index}: {exc}") from exc

    return out


def load_frames_from_json(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SIIValidationError(f"Input file not found: {path}")
    if not p.is_file():
        raise SIIValidationError(f"Input path is not a file: {path}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SIIValidationError(f"Invalid JSON input: {path}") from exc
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise SIIValidationError("JSON input must be an object or an array of objects")
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SIIValidationError("Each JSON item must be an object")
        out.append(item)
    return out


def load_frames_from_csv(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise SIIValidationError(f"Input file not found: {path}")
    if not p.is_file():
        raise SIIValidationError(f"Input path is not a file: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        raise SIIValidationError(f"Failed to read CSV input: {path}") from exc
    parsed = frames_from_csv(text)
    out: list[dict[str, Any]] = []
    for f in parsed:
        out.append(
            {
                "timestamp": f.timestamp,
                "site_id": f.site_id,
                "asset_id": f.asset_id,
                "sensor_values": dict(f.sensor_values),
            }
        )
    return out

