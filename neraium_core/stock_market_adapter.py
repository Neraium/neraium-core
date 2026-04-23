from __future__ import annotations

from typing import Any, Mapping

from .pipeline import DEFAULT_CUSTOMER_ID, build_frame, coerce_float


def build_stock_frame(
    *,
    timestamp: Any,
    ticker: str,
    row_dict: Mapping[str, Any],
    customer_id: str = DEFAULT_CUSTOMER_ID,
    site_id: str = "market",
) -> dict[str, Any]:
    """
    Build a Neraium internal telemetry frame from a market bar/tick payload.

    This adapter is intentionally conservative:
    - Only numeric fields are forwarded as sensor values.
    - Non-numeric fields are ignored to keep the ingest contract stable.
    """

    sensor_values: dict[str, float] = {}
    for key, value in dict(row_dict or {}).items():
        try:
            numeric = coerce_float(value, sensor_name=str(key))
        except ValueError:
            numeric = None
        if numeric is None:
            continue
        sensor_values[str(key)] = float(numeric)

    return build_frame(
        timestamp=timestamp,
        customer_id=customer_id,
        site_id=site_id,
        asset_id=ticker,
        sensor_values=sensor_values,
    )

