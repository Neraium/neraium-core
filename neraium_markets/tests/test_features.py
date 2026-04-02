from config import DATE_COLUMN
from neraium.alignment import align_close_series
from neraium.data_loader import load_all_assets
from neraium.features import build_feature_table


def _feature_table():
    data = load_all_assets()
    aligned = align_close_series(data)
    return aligned, build_feature_table(aligned)


def test_return_columns_exist():
    _, features = _feature_table()
    assert "spy_ret_1d" in features.columns
    assert "spy_ret_5d" in features.columns


def test_volatility_columns_exist():
    _, features = _feature_table()
    for asset in ["spy", "qqq", "iwm"]:
        assert f"{asset}_vol_10d" in features.columns
        assert f"{asset}_vol_20d" in features.columns


def test_breadth_feature_exists():
    _, features = _feature_table()
    assert "breadth_pct_above_20dma" in features.columns


def test_sector_features_exist():
    _, features = _feature_table()
    assert "sector_dispersion_1d" in features.columns
    assert "sector_concentration_top2" in features.columns


def test_cross_asset_features_exist():
    _, features = _feature_table()
    for col in [
        "vix_ret_1d",
        "dxy_ret_1d",
        "gold_ret_1d",
        "oil_ret_1d",
        "us2y_ret_1d",
        "us10y_ret_1d",
        "rates_2s10s",
        "risk_off_proxy",
    ]:
        assert col in features.columns


def test_feature_table_sorted():
    _, features = _feature_table()
    assert features[DATE_COLUMN].is_monotonic_increasing


def test_timestamp_preserved():
    aligned, features = _feature_table()
    assert DATE_COLUMN in features.columns
    assert aligned[DATE_COLUMN].equals(features[DATE_COLUMN])


def test_no_duplicate_columns():
    _, features = _feature_table()
    assert features.columns.is_unique
