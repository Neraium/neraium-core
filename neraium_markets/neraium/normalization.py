"""Day 13: normalize connector output to canonical OHLCV schema."""

from __future__ import annotations

import pandas as pd

CANONICAL_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

# Optional alias map (lowercase target -> accepted lowercase synonyms)
_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "date", "time", "datetime", "ts"),
    "open": ("open", "o"),
    "high": ("high", "h"),
    "low": ("low", "l"),
    "close": ("close", "c", "adj close", "adj_close"),
    "volume": ("volume", "vol", "v"),
}


def _first_matching_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def normalize_market_dataframe(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Map columns to canonical names, parse timestamps, coerce numerics, sort, drop duplicate timestamps.

    ``source_name`` is recorded for debugging only (audit uses connector metadata).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))

    work = df.copy()
    work.columns = [str(c).strip().lower() for c in work.columns]

    out: dict[str, pd.Series] = {}
    for canon_col, aliases in _COLUMN_ALIASES.items():
        src = _first_matching_column(work, aliases)
        if src is None:
            continue
        out[canon_col] = work[src]

    # If timestamp missing, fail later in quality checks
    if "timestamp" in out:
        ts = pd.to_datetime(out["timestamp"], utc=False, errors="coerce")
        out["timestamp"] = ts

    for col in ("open", "high", "low", "close", "volume"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    frame = pd.DataFrame(out)
    for c in CANONICAL_COLUMNS:
        if c not in frame.columns:
            frame[c] = pd.NA

    frame = frame[list(CANONICAL_COLUMNS)]
    frame = frame.sort_values("timestamp", ascending=True).reset_index(drop=True)
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame.attrs["normalization_source"] = source_name
    return frame


def normalize_many_assets(data: dict[str, pd.DataFrame], source_name: str) -> dict[str, pd.DataFrame]:
    """Normalize each asset DataFrame; keys are lowercased asset names."""
    return {k.lower(): normalize_market_dataframe(v, source_name) for k, v in data.items()}
