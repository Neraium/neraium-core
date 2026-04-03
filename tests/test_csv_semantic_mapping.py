from __future__ import annotations

import pytest

from neraium_core.csv_mapping import (
    SemanticColumnMapping,
    infer_semantic_mapping,
    parse_csv_sample_for_mapping,
    resolve_mapping,
    validate_mapping,
)
from neraium_core.pipeline import parse_csv_text
from neraium_core.pipeline import run_csv_ingestion_pipeline


def test_infer_arbitrary_headers() -> None:
    headers = ["recorded_at", "plant", "unit_tag", "pressure_bar", "vibration_rms"]
    mapping, issues, _dbg = infer_semantic_mapping(headers, sample_rows=None)
    assert mapping is not None
    assert mapping.timestamp == "recorded_at"
    assert mapping.asset_id == "unit_tag"
    assert mapping.site_id == "plant"
    assert "pressure_bar" in mapping.sensor_columns
    assert validate_mapping(mapping, headers) == []


def test_infer_without_site_uses_default_later() -> None:
    headers = ["Time", "Machine", "s1", "s2"]
    rows = [{"Time": "2026-01-01T00:00:00+00:00", "Machine": "M1", "s1": "1", "s2": "2"}]
    mapping, _issues, _dbg = infer_semantic_mapping(headers, sample_rows=rows)
    assert mapping is not None
    assert mapping.site_id is None
    assert validate_mapping(mapping, headers) == []


def test_parse_csv_text_with_explicit_mapping() -> None:
    csv_text = "t_col,a_col,x\n2026-01-01T00:00:00+00:00,A1,3.5\n"
    m = {
        "timestamp": "t_col",
        "asset_id": "a_col",
        "site_id": None,
        "sensor_columns": ["x"],
    }
    frames = parse_csv_text(csv_text, column_mapping=m)
    assert len(frames) == 1
    assert frames[0]["timestamp"].startswith("2026-01-01")
    assert frames[0]["asset_id"] == "A1"
    assert frames[0]["site_id"] == "default-site"
    assert frames[0]["sensor_values"]["x"] == 3.5


def test_parse_csv_sample_for_mapping() -> None:
    text = "a,b\n1,2\n3,4\n"
    headers, rows = parse_csv_sample_for_mapping(text, max_rows=5)
    assert headers == ["a", "b"]
    assert len(rows) == 2


def test_validate_mapping_requires_sensors() -> None:
    m = SemanticColumnMapping.from_dict(
        {"timestamp": "t", "asset_id": "a", "site_id": None, "sensor_columns": []}
    )
    errs = validate_mapping(m, ["t", "a", "x"])
    assert any("at least one sensor" in e.lower() for e in errs)


def test_resolve_mapping_requires_confirmation_on_ambiguous_headers() -> None:
    headers = ["timestamp", "time", "recorded_at", "asset", "value"]
    with pytest.raises(ValueError, match="ambiguous_mapping"):
        resolve_mapping(headers, None)


def test_csv_pipeline_generates_canonical_records() -> None:
    csv_text = (
        "when,unit,site,temp,pressure\n"
        "2026-01-01T00:00:00+00:00,A-1,plant-1,10.2,50.1\n"
    )
    result = run_csv_ingestion_pipeline(csv_text, customer_id="cust-a")
    assert result.mapping is not None
    assert len(result.canonical_records) == 1
    rec = result.canonical_records[0]
    assert rec.asset_id == "A-1"
    assert rec.site_id == "plant-1"
    assert rec.timestamp.startswith("2026-01-01T00:00:00")
    assert set(rec.signals.keys()) == {"temp", "pressure"}


def test_csv_pipeline_explicit_mapping_override() -> None:
    csv_text = "t,a,sig\n2026-01-01T00:00:00+00:00,asset-x,3.2\n"
    override = {"timestamp": "t", "asset_id": "a", "site_id": None, "sensor_columns": ["sig"]}
    result = run_csv_ingestion_pipeline(csv_text, customer_id="cust-a", column_mapping=override)
    assert result.mapping is not None
    assert len(result.issues) == 0
    assert result.canonical_records[0].signals["sig"] == 3.2


def test_csv_pipeline_validation_failures_and_malformed_rows() -> None:
    csv_text = "t,a,sig\n2026-01-01T00:00:00+00:00,asset-x,1.0,EXTRA\n"
    result = run_csv_ingestion_pipeline(csv_text, customer_id="cust-a")
    assert any(i.code == "malformed_row" for i in result.issues)


def test_csv_pipeline_mixed_rows_only_emits_valid_canonical_records() -> None:
    csv_text = (
        "timestamp,asset,s1\n"
        "2026-01-01T00:00:00+00:00,ok-a,1.1\n"
        "bad-ts,ok-b,1.2\n"
        "2026-01-01T00:00:02+00:00,ok-c,bad\n"
    )
    result = run_csv_ingestion_pipeline(csv_text, customer_id="cust-a")
    assert len(result.canonical_records) == 1
    codes = {i.code for i in result.issues}
    assert "invalid_timestamp" in codes
    assert "non_numeric_signal" in codes


def test_csv_pipeline_fd001_style_columns_with_explicit_mapping() -> None:
    csv_text = (
        "recorded_at,cycle,unit,setting_1,setting_2,setting_3,s1,s2,s3,s4,s5\n"
        "2026-01-01T00:00:01+00:00,1,fd001_unit_001,0.1,0.2,0.3,10.0,11.0,12.0,13.0,14.0\n"
        "2026-01-01T00:00:02+00:00,2,fd001_unit_001,0.1,0.2,0.3,10.4,11.4,12.4,13.4,14.4\n"
    )
    override = {
        "timestamp": "recorded_at",
        "asset_id": "unit",
        "site_id": None,
        "sensor_columns": ["cycle", "setting_1", "setting_2", "setting_3", "s1", "s2", "s3", "s4", "s5"],
    }
    result = run_csv_ingestion_pipeline(csv_text, customer_id="cust-fd001", column_mapping=override)
    assert len(result.issues) == 0
    assert len(result.canonical_records) == 2
    assert result.canonical_records[0].asset_id == "fd001_unit_001"
    assert result.canonical_records[0].signals["s1"] == 10.0
    assert result.canonical_records[1].signals["s5"] == 14.4


def test_csv_pipeline_fd001_style_row_reports_signal_error_with_row_index() -> None:
    csv_text = (
        "recorded_at,unit,s1,s2,s3\n"
        "2026-01-01T00:00:01+00:00,fd001_unit_099,1.0,2.0,3.0\n"
        "2026-01-01T00:00:02+00:00,fd001_unit_099,1.2,NaNish,3.2\n"
    )
    override = {
        "timestamp": "recorded_at",
        "asset_id": "unit",
        "site_id": None,
        "sensor_columns": ["s1", "s2", "s3"],
    }
    result = run_csv_ingestion_pipeline(csv_text, customer_id="cust-fd001", column_mapping=override)
    assert len(result.canonical_records) == 2
    assert any(i.code == "non_numeric_signal" and i.row == 3 and i.column == "s2" for i in result.issues)
