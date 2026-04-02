from config import ASSETS, DATE_COLUMN
from neraium.alignment import align_close_series
from neraium.data_loader import load_all_assets


def test_alignment_columns_present():
    data = load_all_assets()
    merged = align_close_series(data)
    expected = {DATE_COLUMN, *ASSETS}
    assert expected.issubset(set(merged.columns))


def test_alignment_sorted_ascending():
    data = load_all_assets()
    merged = align_close_series(data)
    assert merged[DATE_COLUMN].is_monotonic_increasing
