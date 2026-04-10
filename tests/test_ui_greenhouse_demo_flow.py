from __future__ import annotations

from ui.app import create_app_state
from ui.config import UIConfig
from ui.core_integration import build_system_state
from ui.demo_data import load_greenhouse_demo_records
from ui.reasoning import build_reasoning_context


REQUIRED_UI_KEYS = {
    "timestamp",
    "site_id",
    "asset_id",
    "state",
    "regime_name",
    "structural_drift_score",
    "relational_stability_score",
    "system_health",
    "confidence_score",
}


def test_greenhouse_demo_adapter_matches_ui_contract() -> None:
    rows = load_greenhouse_demo_records(limit=80)
    assert len(rows) == 80
    assert REQUIRED_UI_KEYS.issubset(set(rows[0].keys()))


def test_greenhouse_demo_replay_rows_are_time_ordered() -> None:
    rows = load_greenhouse_demo_records(limit=60)
    timestamps = [str(row["timestamp"]) for row in rows]
    assert timestamps == sorted(timestamps)


def test_greenhouse_overlay_summary_composition_path_stays_intact() -> None:
    rows = load_greenhouse_demo_records(limit=40)
    state = build_system_state(rows, config=UIConfig())
    context = build_reasoning_context(state, rows)

    assert context["chart_replay_summary"] is not None
    assert context["chart_replay_summary"]["trajectory_points"] >= 1
    assert "current_position" in context["chart_replay_summary"]

    app_state = create_app_state(rows)
    summary = app_state["summary"]
    assert summary["timestamp"]
    assert summary["site_id"]
    assert summary["asset_id"]
    assert summary["regime"]
    story = summary.get("replay_story") or {}
    assert "state_transitions" in story
    assert "drift_trend" in story



def test_create_app_state_defaults_to_suppress_when_empty() -> None:
    state = create_app_state([])
    assert set(state.keys()) == {"summary", "gate_decision", "reasoning_context", "realtime"}
    assert state["gate_decision"]["decision"] == "SUPPRESS"
    assert state["reasoning_context"]["gate_decision"]["decision"] == "SUPPRESS"
    assert state["summary"]["timestamp"] is None
