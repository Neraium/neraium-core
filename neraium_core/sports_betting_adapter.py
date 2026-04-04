from __future__ import annotations

from typing import Any


def build_betting_frame(*, timestamp: object, market_id: str, row_dict: dict[str, Any]) -> dict[str, Any]:
    sensor_values: dict[str, float] = {}
    for key, value in row_dict.items():
        if isinstance(value, bool) or value is None:
            continue
        try:
            sensor_values[key] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        "timestamp": float(timestamp),
        "site_id": "sportsbook",
        "asset_id": market_id,
        "sensor_values": sensor_values,
    }
