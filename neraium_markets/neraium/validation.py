"""Validate loaded market DataFrames."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from neraium.schemas import OHLCVRow  # noqa: E402


def validate_dataframe(name: str, df: pd.DataFrame) -> list[str]:
    """
    Return a list of human-readable validation errors (empty if valid).
    """
    errors: list[str] = []
    if df is None or df.empty:
        errors.append(f"{name}: dataframe is empty")
        return errors

    for col in config.REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"{name}: missing required column {col!r}")

    if errors:
        return errors

    ts = df[config.DATE_COLUMN]
    if ts.isna().any():
        errors.append(f"{name}: null timestamp(s) at index {list(ts[ts.isna()].index[:5])}")

    dup = ts.duplicated()
    if dup.any():
        errors.append(f"{name}: duplicate timestamp(s), count={int(dup.sum())}")

    if not ts.is_monotonic_increasing:
        errors.append(f"{name}: timestamps are not sorted ascending")

    close = df[config.PRICE_COLUMN]
    if not pd.api.types.is_numeric_dtype(close):
        errors.append(f"{name}: {config.PRICE_COLUMN!r} is not numeric")
    else:
        coerced = pd.to_numeric(close, errors="coerce")
        if coerced.isna().any():
            errors.append(f"{name}: non-numeric {config.PRICE_COLUMN!r} value(s)")

    if not errors:
        try:
            r = df.iloc[0]
            OHLCVRow.model_validate(
                {
                    "timestamp": pd.Timestamp(r[config.DATE_COLUMN]).to_pydatetime(),
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r[config.PRICE_COLUMN]),
                    "volume": float(r["volume"]),
                }
            )
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{name}: sample row failed pydantic OHLCVRow check: {exc}")

    return errors


def validate_all(data: dict[str, pd.DataFrame]) -> list[str]:
    """Validate every asset in the mapping."""
    errors: list[str] = []
    for name, df in sorted(data.items()):
        errors.extend(validate_dataframe(name, df))
    return errors
