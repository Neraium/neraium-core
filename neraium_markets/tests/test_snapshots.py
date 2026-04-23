"""Day 14: snapshots."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neraium.snapshots import load_latest_snapshot, save_input_snapshot, save_output_snapshot


def test_input_snapshot_written(tmp_path: Path) -> None:
    data = {"spy": pd.DataFrame({"timestamp": ["2024-01-01"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]})}
    meta = save_input_snapshot(data, tmp_path, "r1", "daily")
    assert "spy" in meta["files"]
    assert Path(meta["files"]["spy"]).is_file()


def test_output_snapshot_written(tmp_path: Path) -> None:
    ms = pd.DataFrame({"timestamp": ["2024-01-01"], "x": [1]})
    meta = save_output_snapshot({"market_state": ms}, tmp_path, "r2")
    assert "market_state" in meta["files"]


def test_load_latest_snapshot(tmp_path: Path) -> None:
    save_output_snapshot(
        {"market_state": pd.DataFrame({"timestamp": ["2024-01-02"], "a": [2]})},
        tmp_path,
        "older",
    )
    save_output_snapshot(
        {"market_state": pd.DataFrame({"timestamp": ["2024-01-03"], "a": [3]})},
        tmp_path,
        "newer",
    )
    loaded = load_latest_snapshot(str(tmp_path), "market_state")
    assert loaded is not None
    assert len(loaded) == 1
    assert str(loaded["timestamp"].iloc[0]).startswith("2024-01-03")
