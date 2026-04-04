from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _to_iso_timestamp(timestamp: object) -> str:
    if isinstance(timestamp, datetime):
        dt = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return datetime.fromtimestamp(0.0, tz=timezone.utc).isoformat()


def build_stock_frame(*, timestamp: object, ticker: str, row_dict: dict[str, Any]) -> dict[str, Any]:
    sensor_values: dict[str, float] = {}
    sensor_quality: dict[str, str] = {}
    for key, value in row_dict.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        sensor_values[key] = numeric
        sensor_quality[key] = "ok"

    return {
        "timestamp": _to_iso_timestamp(timestamp),
        "customer_id": "default-customer",
        "site_id": "market",
        "asset_id": str(ticker).upper(),
        "sensor_values": sensor_values,
        "sensor_quality": sensor_quality,
        "aligned": [],
        "anomaly": False,
    }
