"""Tests for validation helpers."""

from __future__ import annotations

import pandas as pd

import config
from neraium.validation import validate_all, validate_dataframe


def _valid_minimal_df():
    ts = pd.date_range("2024-01-02", periods=5, freq="B")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [1, 2, 3, 4, 5],
            "high": [2, 3, 4, 5, 6],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.5, 2.5, 3.5, 4.5, 5.5],
            "volume": [100, 200, 300, 400, 500],
        }
    )


def test_validate_clean_dataframe_no_errors():
    df = _valid_minimal_df()
    assert validate_dataframe("test", df) == []


def test_validate_missing_required_column():
    df = _valid_minimal_df().drop(columns=["volume"])
    errs = validate_dataframe("x", df)
    assert any("missing required column" in e for e in errs)


def test_validate_duplicate_timestamp():
    df = _valid_minimal_df()
    df = pd.concat([df.iloc[:1], df], ignore_index=True)
    errs = validate_dataframe("x", df)
    assert any("duplicate timestamp" in e for e in errs)


def test_validate_unsorted_timestamps():
    df = _valid_minimal_df()
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    errs = validate_dataframe("x", df)
    assert any("not sorted ascending" in e for e in errs)


def test_validate_all_collects_multiple_assets():
    bad = _valid_minimal_df().drop(columns=["close"])
    good = _valid_minimal_df()
    errs = validate_all({"a": bad, "b": good})
    assert len(errs) >= 1
    assert any("a:" in e for e in errs)
