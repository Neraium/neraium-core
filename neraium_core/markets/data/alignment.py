"""Series alignment and shared clock operations."""

from __future__ import annotations

import pandas as pd


def align_to_shared_clock(df: pd.DataFrame) -> pd.DataFrame:
    """Align all series to the shared timestamp index with interpolation."""
    aligned = df.sort_index().copy()
    aligned = aligned[~aligned.index.duplicated(keep="last")]
    return aligned.ffill().bfill()
