"""Tests for Day 11 realtime cycle."""

from __future__ import annotations

from pathlib import Path

from neraium.realtime import run_realtime_cycle, write_realtime_snapshot_json


def test_realtime_cycle_returns_snapshot() -> None:
    cyc = run_realtime_cycle(Path(__file__).resolve().parents[1], emit_evidence_log=False)
    assert "snapshot" in cyc
    assert cyc.get("pipeline_success") is True
    snap = cyc.get("snapshot")
    assert isinstance(snap, dict)
    for k in (
        "timestamp",
        "market_regime_label",
        "market_confidence_score",
        "market_warning_level",
        "path_adjusted_market_action",
        "market_explanation",
    ):
        assert k in snap


def test_write_realtime_snapshot_json(tmp_path: Path) -> None:
    p = tmp_path / "snap.json"
    write_realtime_snapshot_json({"timestamp": "2024-01-01T00:00:00", "x": 1}, p)
    assert p.is_file()
    assert "timestamp" in p.read_text(encoding="utf-8")
