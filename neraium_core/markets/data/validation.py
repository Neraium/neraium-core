"""Validation checks for market telemetry integrity."""

from __future__ import annotations

import pandas as pd


def validate_market_frame(df: pd.DataFrame) -> dict[str, float | int]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    duplicates = int(df.index.duplicated().sum())
    missing = int(df.isna().sum().sum())
    completeness = 1.0 - (missing / float(df.shape[0] * max(df.shape[1], 1)))
    return {"duplicate_timestamps": duplicates, "missing_values": missing, "completeness": completeness}
