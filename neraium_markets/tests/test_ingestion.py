"""Day 13: ingestion orchestrator."""

from __future__ import annotations

from pathlib import Path

from neraium.ingestion import build_ingestion_summary, ingest_market_data
from neraium.settings import load_settings, with_overrides


def test_ingestion_returns_normalized_assets() -> None:
    s = with_overrides(
        load_settings("dev"),
        source_type="mock_api",
        mock_api_fixture_dir=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "mock_api",
        reject_invalid_assets=False,
    )
    assets = ["spy", "qqq"]
    data, summary = ingest_market_data(assets, "daily", s)
    assert set(data.keys()) == {"spy", "qqq"}
    assert "timestamp" in data["spy"].columns


def test_ingestion_summary_exists() -> None:
    s = with_overrides(
        load_settings("dev"),
        source_type="csv",
        reject_invalid_assets=False,
    )
    assets = ["spy"]
    _, summary = ingest_market_data(assets, "daily", s)
    for k in (
        "source_type",
        "assets_requested",
        "assets_loaded",
        "assets_valid",
        "assets_rejected",
        "quality_status_counts",
        "source_metadata",
    ):
        assert k in summary


def test_invalid_assets_can_be_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "timestamp,open,high,low,close,volume\n"
        + "\n".join(
            f"2024-01-{i:02d},1,1,1,1,1"
            for i in range(1, 6)
        )
        + "\n",
        encoding="utf-8",
    )
    s = with_overrides(
        load_settings("dev"),
        source_type="csv",
        data_dir=tmp_path,
        reject_invalid_assets=True,
        active_assets=("bad",),
        source_quality_thresholds={"min_rows": 100.0},
    )
    data, summary = ingest_market_data(["bad"], "daily", s)
    assert "bad" not in data
    assert "bad" in summary.get("assets_rejected", [])


def test_build_ingestion_summary_shape() -> None:
    sm = build_ingestion_summary(
        source_type="csv",
        assets_requested=["a"],
        assets_loaded=["a"],
        assets_valid=["a"],
        assets_rejected=[],
        quality_rows=[{"asset": "a", "quality_status": "good", "valid_for_pipeline": True}],
        source_metadata={"x": 1},
    )
    assert sm["quality_status_counts"]["good"] == 1
