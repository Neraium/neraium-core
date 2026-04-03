from __future__ import annotations

import math
from numbers import Number
from typing import Any

from neraium_core.engine_ingress import DEFAULT_CUSTOMER_ID, EngineIngressFrame, normalize_timestamp

PREFERRED_MARKET_FIELDS = ("open", "high", "low", "close", "volume", "value")


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


def build_stock_frame(timestamp: Any, ticker: Any, row_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a market row into the common frame contract expected by engines."""
    sensors: dict[str, float] = {}
    ordered_fields = list(PREFERRED_MARKET_FIELDS)
    ordered_fields.extend(k for k in row_dict.keys() if k not in PREFERRED_MARKET_FIELDS)

    for key in ordered_fields:
        if key not in row_dict:
            continue
        numeric = _to_numeric(row_dict.get(key))
        if numeric is not None:
            sensors[str(key)] = numeric

    frame = EngineIngressFrame(
        timestamp=normalize_timestamp(timestamp),
        customer_id=DEFAULT_CUSTOMER_ID,
        site_id="market",
        asset_id=str(ticker),
        sensor_values=sensors,
        sensor_quality={name: "ok" for name in sensors},
    )
    return frame.to_dict()


__all__ = ["build_stock_frame"]
