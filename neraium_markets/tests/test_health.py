"""Tests for Day 11 health."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DATE_COLUMN
from neraium.health import compute_system_health, log_health_snapshot


def test_health_snapshot_fields() -> None:
    df = pd.DataFrame({DATE_COLUMN: pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")})
    h = compute_system_health(df, pipeline_success=True, recent_alert_count=0)
    assert "data_freshness_ok" in h
    assert "missing_data_rate" in h
    assert "latest_timestamp" in h
    assert "pipeline_success" in h
    assert "health_status" in h
    assert h["pipeline_success"] is True


def test_health_log_created(tmp_path: Path) -> None:
    p = tmp_path / "health.jsonl"
    log_health_snapshot({"health_status": "healthy", "x": 1}, p)
    assert p.is_file()
    assert "healthy" in p.read_text(encoding="utf-8")
