from __future__ import annotations

from typing import Iterable

import pandas as pd

TIMESTAMP_ALIASES = ("timestamp", "time", "date", "datetime", "ts")
TICKER_ALIASES = ("ticker", "symbol", "asset", "asset_id", "instrument")


def _resolve_column(columns: Iterable[str], aliases: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Column {explicit!r} not found in dataset")
        return explicit

    lowered = {str(c).lower(): str(c) for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    raise ValueError(f"Could not resolve required column from aliases: {aliases}")


def load_market_data(
    csv_path: str,
    *,
    timestamp_column: str | None = None,
    ticker_column: str | None = None,
) -> pd.DataFrame:
    """Load and normalize market CSV data for ticker-wise engine processing."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    resolved_timestamp = _resolve_column(df.columns, TIMESTAMP_ALIASES, timestamp_column)
    resolved_ticker = _resolve_column(df.columns, TICKER_ALIASES, ticker_column)

    df = df.rename(columns={resolved_timestamp: "timestamp", resolved_ticker: "ticker"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp", "ticker"]).copy()

    for col in df.columns:
        if col in {"timestamp", "ticker"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "value" not in df.columns and "close" in df.columns:
        df["value"] = df["close"]

    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return df


__all__ = ["load_market_data", "TIMESTAMP_ALIASES", "TICKER_ALIASES"]
