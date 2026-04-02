from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel

from config import DATE_COLUMN, PRICE_COLUMN


class ValidationResult(BaseModel):
    name: str
    errors: list[str]


REQUIRED_COLUMNS = {DATE_COLUMN, "open", "high", "low", PRICE_COLUMN, "volume"}


def validate_dataframe(name: str, df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        errors.append(f"{name}: missing required columns: {sorted(missing)}")
        return errors

    if df[DATE_COLUMN].isna().any():
        errors.append(f"{name}: null timestamps found")

    if df[DATE_COLUMN].duplicated().any():
        errors.append(f"{name}: duplicate timestamps found")

    if not df[DATE_COLUMN].is_monotonic_increasing:
        errors.append(f"{name}: timestamps are not sorted ascending")

    if not is_numeric_dtype(df[PRICE_COLUMN]):
        errors.append(f"{name}: close column is not numeric")

    return errors


def validate_all(data: dict[str, pd.DataFrame]) -> list[str]:
    all_errors: list[str] = []
    for name, df in data.items():
        result = ValidationResult(name=name, errors=validate_dataframe(name, df))
        all_errors.extend(result.errors)
    return all_errors
