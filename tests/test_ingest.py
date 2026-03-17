import pytest

from neraium_core.ingest import normalize_rest_payload, parse_csv_text


def test_invalid_sensor_values_remain_none() -> None:
    frame = normalize_rest_payload(
        {
            "timestamp": "2026-01-01T01:02:03",
            "site_id": "site a",
            "asset_id": "asset/a",
            "sensor_values": {"ok": "1.23", "bad": "abc", "missing": ""},
        }
    )

    assert frame["sensor_values"]["ok"] == pytest.approx(1.23)
    assert frame["sensor_values"]["bad"] is None
    assert frame["sensor_values"]["missing"] is None
    assert frame["timestamp"].endswith("+00:00")
    assert frame["site_id"] == "site_a"
    assert frame["asset_id"] == "asset_a"


def test_csv_row_validation_error_contains_row_number() -> None:
    csv_text = "timestamp,site_id,asset_id,,s1\n2026-01-01T00:00:00Z,site,asset,1,2\n"
    with pytest.raises(ValueError, match="Invalid CSV row 2"):
        parse_csv_text(csv_text)
