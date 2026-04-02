import pandas as pd

from neraium.validation import validate_dataframe


def test_validation_catches_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "open": [1.0, 1.1],
            "high": [1.2, 1.3],
            "low": [0.9, 1.0],
            "close": [1.1, 1.2],
            "volume": [100, 110],
        }
    )
    errors = validate_dataframe("dup", df)
    assert any("duplicate timestamps" in e for e in errors)


def test_validation_catches_missing_required_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01"]),
            "open": [1.0],
            "high": [1.2],
            "low": [0.9],
            "volume": [100],
        }
    )
    errors = validate_dataframe("missing", df)
    assert any("missing required columns" in e for e in errors)
