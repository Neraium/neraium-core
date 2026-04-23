"""Day 14: state store."""

from __future__ import annotations

from pathlib import Path

from neraium.settings import load_settings
from neraium.state_store import build_default_state, load_pipeline_state, save_pipeline_state


def test_save_and_load_pipeline_state(tmp_path: Path) -> None:
    p = tmp_path / "st.json"
    d = {"a": 1, "b": [1, 2]}
    save_pipeline_state(d, p)
    loaded = load_pipeline_state(p)
    assert loaded == d


def test_default_state_created() -> None:
    s = load_settings("dev")
    st = build_default_state(s)
    assert "last_successful_run_id" in st
    assert st["source_type"] == s.source_type
    assert "latest_timestamp_by_asset" in st
