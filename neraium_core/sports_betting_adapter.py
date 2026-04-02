from __future__ import annotations

import math
from numbers import Number
from typing import Any


def _to_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Number):
        numeric = float(value)
    else:
        text = str(value).strip() if value is not None else ""
        if text == "":
            return None
        try:
            numeric = float(text)
        except (TypeError, ValueError):
            return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def build_betting_frame(timestamp: Any, market_id: Any, row_dict: dict[str, Any]) -> dict[str, Any]:
    sensors: dict[str, float] = {}
    for key, value in row_dict.items():
        numeric = _to_numeric(value)
        if numeric is not None:
            sensors[str(key)] = numeric

    return {
        "timestamp": float(timestamp),
        "site_id": "sportsbook",
        "asset_id": str(market_id),
        "sensor_values": sensors,
    }


__all__ = ["build_betting_frame"]
