from __future__ import annotations

import pandas as pd

from config import DATE_COLUMN, PRICE_COLUMN


def align_close_series(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join all asset close series on timestamp."""
    merged: pd.DataFrame | None = None

    for asset, df in data.items():
        subset = df[[DATE_COLUMN, PRICE_COLUMN]].copy()
        subset = subset.rename(columns={PRICE_COLUMN: asset})

        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on=DATE_COLUMN, how="outer")

    if merged is None:
        return pd.DataFrame(columns=[DATE_COLUMN])

    return merged.sort_values(DATE_COLUMN, ascending=True).reset_index(drop=True)
