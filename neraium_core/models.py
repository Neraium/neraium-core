from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class SignalDefinition(BaseModel):
    name: str
    dtype: Literal["float64", "int64"]
    unit: str
    required_for_scoring: bool


class SystemDefinition(BaseModel):
    system_id: str
    schema_version: str
    raw_sample_period_seconds: int
    inference_window_seconds: int
    max_forward_fill_windows: int
    max_missing_signal_fraction: float
    signals: list[SignalDefinition]
    vector_order: list[str]


class TelemetryPayload(BaseModel):
    system_id: str
    timestamp: datetime
    signals: dict[str, float | int | None]


class StructuralResult(BaseModel):
    drift: float
    spectral: float
    directional: float
    entropy: float
    early_warning: float
    subsystem: float
    composite_score: float
