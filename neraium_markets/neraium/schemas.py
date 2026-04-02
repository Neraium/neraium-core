"""Pydantic models for row-level validation (sample checks)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class OHLCVRow(BaseModel):
    """Expected shape of one CSV row after type coercion."""

    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
