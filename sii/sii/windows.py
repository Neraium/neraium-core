from __future__ import annotations

import pandas as pd


def sliding_windows(df: pd.DataFrame, size: int, step: int = 1) -> list[pd.DataFrame]:
    if len(df) < size:
        return []
    windows: list[pd.DataFrame] = []
    for start in range(0, len(df) - size + 1, step):
        windows.append(df.iloc[start : start + size].copy())
    return windows
