from config import DATE_COLUMN
from neraium.data_loader import load_all_assets, load_asset_csv


def test_load_asset_csv_has_timestamp_column():
    df = load_asset_csv("spy")
    assert DATE_COLUMN in df.columns
    assert df[DATE_COLUMN].is_monotonic_increasing


def test_load_all_assets_returns_dict():
    data = load_all_assets()
    assert isinstance(data, dict)
    assert "spy" in data
    assert len(data) >= 17
