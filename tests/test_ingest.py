import pytest

from neraium_core.ingest import (
    build_frame,
    normalize_rest_payload,
    normalize_timestamp,
    parse_csv_text,
)


def test_invalid_sensor_values_remain_missing_not_zero() -> None:
    frame = normalize_rest_payload(
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "site_id": "a",
            "asset_id": "b",
            "sensor_values": {"temp": "bad", "pressure": ""},
        }
    )
    assert frame.sensor_values["temp"] is None
    assert frame.sensor_values["pressure"] is None


def test_malformed_timestamp_raises_useful_error() -> None:
    with pytest.raises(ValueError, match="invalid timestamp"):
        normalize_timestamp("not-a-time")


def test_parse_csv_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="CSV missing required columns"):
        parse_csv_text("timestamp,site_id,temp\n2026-01-01T00:00:00Z,s1,2.0\n")


def test_parse_csv_preserves_sensor_columns() -> None:
    csv_text = (
        "timestamp,site_id,asset_id,temp,vibration\n"
        "2026-01-01T00:00:00Z,s1,a1,4.2,0.3\n"
    )
    frames = parse_csv_text(csv_text)
    assert len(frames) == 1
    assert set(frames[0].sensor_values.keys()) == {"temp", "vibration"}


def test_build_frame_marks_quality() -> None:
    frame = build_frame("2026-01-01T00:00:00Z", "s", "a", {"temp": "1", "p": "bad"})
    assert frame.sensor_quality["temp"] == "ok"
    assert frame.sensor_quality["p"] == "missing"
