from neraium_core.ingest import parse_csv_text


def test_parse_csv_text() -> None:
    csv_text = """timestamp,site_id,asset_id,temp\n2026-01-01T00:00:00Z,a,b,42.0\n"""

    frames = parse_csv_text(csv_text)

    assert len(frames) == 1
    assert frames[0]["site_id"] == "a"
    assert frames[0]["sensor_values"]["temp"] == 42.0
