"""Day 13: assess normalized frames before core pipeline validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neraium.normalization import CANONICAL_COLUMNS  # noqa: E402


def _default_thresholds() -> dict[str, float]:
    return {
        "max_null_timestamp_ratio": 0.05,
        "max_null_close_ratio": 0.05,
        "max_duplicate_timestamp_ratio": 0.01,
        "min_rows": 5,
    }


def assess_source_quality(
    df: pd.DataFrame,
    asset: str,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Return quality diagnostics for one normalized asset.

    ``quality_status``: ``good`` | ``degraded`` | ``invalid``.
    """
    issues: list[str] = []
    th = {**_default_thresholds(), **(thresholds or {})}

    if df is None or df.empty:
        return {
            "asset": asset,
            "quality_status": "invalid",
            "issues": ["empty dataframe"],
            "missing_rate": 1.0,
            "duplicate_count": 0,
            "valid_for_pipeline": False,
        }

    n = len(df)
    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        return {
            "asset": asset,
            "quality_status": "invalid",
            "issues": [f"missing columns: {missing_cols}"],
            "missing_rate": 1.0,
            "duplicate_count": 0,
            "valid_for_pipeline": False,
        }

    ts = df["timestamp"]
    null_ts = int(ts.isna().sum())
    null_ts_ratio = float(null_ts) / float(n) if n else 1.0

    close = df["close"]
    null_close = int(close.isna().sum())
    null_close_ratio = float(null_close) / float(n) if n else 1.0

    dup_count = int(df["timestamp"].duplicated().sum()) if "timestamp" in df.columns else 0
    dup_ratio = float(dup_count) / float(n) if n else 0.0

    mono_ok = True
    if n > 1 and df["timestamp"].notna().all():
        mono_ok = bool(df["timestamp"].is_monotonic_increasing)
        if not mono_ok:
            issues.append("timestamps not monotonic increasing")

    coerced_close = pd.to_numeric(df["close"], errors="coerce")
    bad_close = int((coerced_close.isna() & df["close"].notna()).sum())
    if bad_close:
        issues.append(f"non-numeric close values: {bad_close}")

    missing_rate = max(null_ts_ratio, null_close_ratio)

    if null_ts_ratio > th["max_null_timestamp_ratio"]:
        issues.append(f"null timestamp ratio {null_ts_ratio:.4f} > max {th['max_null_timestamp_ratio']}")
    if null_close_ratio > th["max_null_close_ratio"]:
        issues.append(f"null close ratio {null_close_ratio:.4f} > max {th['max_null_close_ratio']}")
    if dup_ratio > th["max_duplicate_timestamp_ratio"]:
        issues.append(f"duplicate timestamp ratio {dup_ratio:.4f} > max {th['max_duplicate_timestamp_ratio']}")
    if n < int(th["min_rows"]):
        issues.append(f"row count {n} < min_rows {int(th['min_rows'])}")

    critical = (
        null_ts_ratio > th["max_null_timestamp_ratio"]
        or null_close_ratio > th["max_null_close_ratio"]
        or dup_ratio > th["max_duplicate_timestamp_ratio"]
        or n < int(th["min_rows"])
        or bad_close > 0
    )

    if critical:
        status = "invalid"
        valid_for = False
    elif issues:
        status = "degraded"
        valid_for = True
    else:
        status = "good"
        valid_for = True

    return {
        "asset": asset,
        "quality_status": status,
        "issues": issues,
        "missing_rate": float(missing_rate),
        "duplicate_count": dup_count,
        "valid_for_pipeline": valid_for,
    }


def assess_many_assets(
    data: dict[str, pd.DataFrame],
    *,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """One row per asset with quality fields."""
    rows = [assess_source_quality(df, asset, thresholds=thresholds) for asset, df in data.items()]
    return pd.DataFrame(rows).sort_values("asset").reset_index(drop=True)
