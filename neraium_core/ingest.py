from __future__ import annotations

from typing import Any, Dict


REQUIRED_FIELDS = {"timestamp", "site_id", "asset_id", "sensor_values"}


def normalize_telemetry_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize telemetry frame, validate numeric signals, and handle missing values."""
    if not isinstance(frame, dict):
        raise ValueError("Telemetry frame must be a dictionary")

    missing = REQUIRED_FIELDS.difference(frame.keys())
    if missing:
        raise ValueError(f"Missing required telemetry fields: {sorted(missing)}")

    raw_sensors = frame.get("sensor_values")
    if not isinstance(raw_sensors, dict) or not raw_sensors:
        raise ValueError("sensor_values must be a non-empty object")

    normalized = {
        "timestamp": str(frame["timestamp"]),
        "site_id": str(frame["site_id"]),
        "asset_id": str(frame["asset_id"]),
        "sensor_values": {},
    }

    for key, value in raw_sensors.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0

        if numeric != numeric:
            numeric = 0.0

        normalized["sensor_values"][str(key)] = numeric

    return normalized
