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
