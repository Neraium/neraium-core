"""Day 13: source quality."""

from __future__ import annotations

import pandas as pd

from neraium.normalization import normalize_market_dataframe
from neraium.source_quality import assess_many_assets, assess_source_quality


def _good_spy() -> pd.DataFrame:
    rows = []
    for i in range(10):
        rows.append(
            {
                "timestamp": f"2024-01-{i+1:02d}",
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.5 + i,
                "volume": 1000 + i,
            }
        )
    return normalize_market_dataframe(pd.DataFrame(rows), "test")


def test_source_quality_good_case() -> None:
    df = _good_spy()
    q = assess_source_quality(df, "spy")
    assert q["valid_for_pipeline"] is True
    assert q["quality_status"] == "good"


def test_source_quality_invalid_case() -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-01"],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
        }
    )
    df = normalize_market_dataframe(raw, "test")
    q = assess_source_quality(df, "bad", thresholds={"max_duplicate_timestamp_ratio": 0.0})
    assert q["quality_status"] == "invalid"
    assert q["valid_for_pipeline"] is False


def test_assess_many_assets_dataframe() -> None:
    df = _good_spy()
    tbl = assess_many_assets({"spy": df})
    assert "asset" in tbl.columns
    assert len(tbl) == 1
