from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TelemetryFrame:
    timestamp: str
    site_id: str
    asset_id: str
    sensor_values: Dict[str, float]
