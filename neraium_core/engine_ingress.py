from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


DEFAULT_SITE_ID = "default-site"
DEFAULT_ASSET_ID = "default-asset"
DEFAULT_CUSTOMER_ID = "default-customer"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(value: Any) -> str:
    """Normalize timestamp-like input into an ISO-8601 UTC string."""
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
            try:
                numeric = float(text)
                if 1e9 <= abs(numeric) <= 1e12:
                    dt = datetime.fromtimestamp(numeric, tz=timezone.utc)
                elif 1e12 < abs(numeric) <= 1e15:
                    dt = datetime.fromtimestamp(numeric / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
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


@dataclass(frozen=True)
class EngineIngressFrame:
    timestamp: str
    customer_id: str
    site_id: str
    asset_id: str
    sensor_values: dict[str, float | None] = field(default_factory=dict)
    sensor_quality: dict[str, str] = field(default_factory=dict)
    aligned: list[Any] = field(default_factory=list)
    anomaly: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "customer_id": self.customer_id,
            "site_id": self.site_id,
            "asset_id": self.asset_id,
            "sensor_values": dict(self.sensor_values),
            "sensor_quality": dict(self.sensor_quality),
            "aligned": list(self.aligned),
            "anomaly": self.anomaly,
        }
