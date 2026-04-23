"""Day 11: system health and freshness tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config import DATE_COLUMN

_DEFAULT_HEALTH_LOG = Path("output") / "health_log.jsonl"

_STALE_HOURS = 72.0


def compute_system_health(
    latest_df: pd.DataFrame,
    *,
    pipeline_success: bool = True,
    recent_alert_count: int = 0,
) -> dict[str, Any]:
    """
    Assess trustworthiness of the latest pipeline output.

    ``latest_df`` should be ``market_state`` (or any frame with ``timestamp``).
    """
    if latest_df.empty or DATE_COLUMN not in latest_df.columns:
        return {
            "data_freshness_ok": False,
            "missing_data_rate": 1.0,
            "latest_timestamp": None,
            "pipeline_success": pipeline_success,
            "recent_alert_count": recent_alert_count,
            "health_status": "failed" if not pipeline_success else "stale",
        }

    ts = pd.to_datetime(latest_df[DATE_COLUMN], utc=True, errors="coerce")
    miss = float(ts.isna().mean())
    latest_ts = ts.max()
    if pd.isna(latest_ts):
        return {
            "data_freshness_ok": False,
            "missing_data_rate": miss,
            "latest_timestamp": None,
            "pipeline_success": pipeline_success,
            "recent_alert_count": recent_alert_count,
            "health_status": "failed",
        }

    lt = pd.Timestamp(latest_ts)
    if lt.tzinfo is None:
        lt = lt.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    age_h = (now - lt).total_seconds() / 3600.0

    fresh = age_h <= _STALE_HOURS and miss < 0.05

    if not pipeline_success:
        status = "failed"
    elif miss >= 0.05 or age_h > _STALE_HOURS:
        status = "stale"
    elif miss > 0.01 or recent_alert_count > 5:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "data_freshness_ok": bool(fresh and pipeline_success),
        "missing_data_rate": float(miss),
        "latest_timestamp": lt.isoformat(),
        "pipeline_success": pipeline_success,
        "recent_alert_count": int(recent_alert_count),
        "health_status": status,
    }


def log_health_snapshot(
    health_snapshot: dict[str, Any],
    output_path: str | Path = _DEFAULT_HEALTH_LOG,
) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(health_snapshot, default=str, ensure_ascii=False) + "\n")
