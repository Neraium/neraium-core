"""Tests for Day 10 evidence logging."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neraium.evidence import log_batch_events, log_signal_event


def test_log_signal_event_creates_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "evidence_log.jsonl"
    row = pd.Series(
        {
            "timestamp": pd.Timestamp("2024-01-01"),
            "market_regime_label": "market_stable",
            "market_confidence_score": 0.7,
            "market_action_posture": "risk_on",
            "path_adjusted_market_action": "risk_on",
            "market_warning_level": "low",
            "scenario_path_label": "unclassified",
            "trajectory_direction": "flat",
            "market_instability_score": 0.4,
            "market_coherence_score": 0.5,
            "path_aware_market_explanation": "test",
        }
    )
    log_signal_event(row, path=p)
    assert p.is_file()
    line = p.read_text(encoding="utf-8").strip()
    rec = json.loads(line)
    assert rec["regime_label"] == "market_stable"
    assert rec["warning_level"] == "low"


def test_log_batch_events_sorted(tmp_path: Path) -> None:
    p = tmp_path / "batch.jsonl"
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-01")],
            "market_regime_label": ["market_mixed", "market_stable"],
            "market_confidence_score": [0.5, 0.6],
            "market_action_posture": ["neutral", "risk_on"],
            "path_adjusted_market_action": ["neutral", "risk_on"],
            "market_warning_level": ["none", "none"],
            "scenario_path_label": ["a", "b"],
            "trajectory_direction": ["flat", "flat"],
            "market_instability_score": [0.5, 0.4],
            "market_coherence_score": [0.5, 0.5],
            "path_aware_market_explanation": ["x", "y"],
        }
    )
    log_batch_events(df, path=p)
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    ts_first = json.loads(lines[0])["timestamp"]
    assert "2024-01-01" in ts_first or "01-01" in ts_first
