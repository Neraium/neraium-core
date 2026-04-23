"""Day 14: incremental processing."""

from __future__ import annotations

import pandas as pd

from neraium.incremental import (
    build_resume_summary,
    detect_new_data,
    filter_to_incremental_window,
    merge_incremental_outputs,
)


def test_detect_new_data() -> None:
    prior = {"latest_timestamp_by_asset": {"spy": "2024-01-01T00:00:00"}}
    data = {
        "spy": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "open": [1, 2],
                "high": [1, 2],
                "low": [1, 2],
                "close": [1, 2],
                "volume": [1, 2],
            }
        )
    }
    out = detect_new_data(data, prior)
    assert out["any_new_data"] is True
    assert out["has_new_data_by_asset"]["spy"] is True


def test_incremental_window_filters_data() -> None:
    prior = {"latest_timestamp_by_asset": {"spy": "2024-01-05T00:00:00"}}
    rows = []
    for i in range(1, 11):
        rows.append(
            {
                "timestamp": pd.Timestamp(f"2024-01-{i:02d}"),
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 1.0,
            }
        )
    data = {"spy": pd.DataFrame(rows)}
    filt = filter_to_incremental_window(data, prior, lookback_buffer=3)
    assert len(filt["spy"]) < len(data["spy"])


def test_merge_incremental_outputs() -> None:
    a = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]), "v": [1, 2]})
    b = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]), "v": [20, 3]})
    m = merge_incremental_outputs(a, b, timestamp_col="timestamp")
    assert len(m) == 3
    assert m["v"].tolist() == [1, 20, 3]


def test_resume_summary_exists() -> None:
    s = build_resume_summary(
        prior_state_found=True,
        incremental_mode_used=False,
        new_data_summary=None,
        cache_hits=0,
        cache_misses=1,
        input_snapshot_written=False,
        output_snapshot_written=False,
        state_updated=False,
    )
    assert "cache_hits" in s
    assert "notes" in s
