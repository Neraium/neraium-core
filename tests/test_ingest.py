import pytest

from neraium_core.ingest import normalize_rest_payload, parse_csv_text


def test_invalid_sensor_values_stay_none():
    payload = {
        "timestamp": "2026-01-01T01:00:00Z",
        "site_id": "  site 1  ",
        "asset_id": "",
        "sensor_values": {"temp": "bad", "pressure": "", "ok": "1.5"},
    }

    frame = normalize_rest_payload(payload, include_sensor_quality=True)

    assert frame["sensor_values"]["temp"] is None
    assert frame["sensor_values"]["pressure"] is None
    assert frame["sensor_values"]["ok"] == 1.5
    assert frame["site_id"] == "site 1"
    assert frame["asset_id"] == "default-asset"


def test_csv_validation_reports_row_number():
    csv_text = "timestamp,site_id,asset_id, ,temp\n2026-01-01T00:00:00Z,s,a,1,2\n"
    with pytest.raises(ValueError, match="row 2"):
        parse_csv_text(csv_text)


def test_csv_includes_sensor_quality_and_timestamp_utc():
    csv_text = "timestamp,site_id,asset_id,temp\n2026-01-01T00:00:00,site,asset,\n"
    rows = parse_csv_text(csv_text, include_sensor_quality=True)
    assert rows[0]["timestamp"].endswith("Z")
    assert rows[0]["sensor_values"]["temp"] is None
    assert rows[0]["sensor_quality"]["temp"] == "missing"
