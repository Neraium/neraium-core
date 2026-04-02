from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - fallback runtime path
    pd = None

TIMESTAMP_ALIASES = ("timestamp", "time", "date", "datetime", "ts")
TICKER_ALIASES = ("ticker", "symbol", "asset", "asset_id", "instrument")
CLOSE_ALIASES = ("close", "last", "settle", "settlement")


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


def _resolve_optional_column(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    lowered = {str(c).lower(): str(c) for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def _parse_timestamp(raw: str) -> datetime | None:
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_numeric(value: str) -> float | None:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _load_market_data_stdlib(
    csv_path: str,
    *,
    timestamp_column: str | None = None,
    ticker_column: str | None = None,
) -> list[dict[str, object]]:
    with Path(csv_path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []
        if not columns:
            return []

        resolved_timestamp = _resolve_column(columns, TIMESTAMP_ALIASES, timestamp_column)
        resolved_ticker = _resolve_column(columns, TICKER_ALIASES, ticker_column)
        close_col = _resolve_optional_column(columns, CLOSE_ALIASES)

        rows: list[dict[str, object]] = []
        for row in reader:
            ts = _parse_timestamp(str(row.get(resolved_timestamp, "")))
            ticker = str(row.get(resolved_ticker, "")).strip()
            if ts is None or ticker == "" or ticker.lower() == "nan":
                continue

            normalized: dict[str, object] = {"timestamp": ts, "ticker": ticker}
            for key, raw in row.items():
                if key in {resolved_timestamp, resolved_ticker}:
                    continue
                numeric = _to_numeric(str(raw) if raw is not None else "")
                normalized[key] = numeric

            if "value" not in normalized and close_col and close_col in normalized:
                normalized["value"] = normalized[close_col]

            rows.append(normalized)

    rows.sort(key=lambda item: (str(item["ticker"]), item["timestamp"]))
    return rows


def load_market_data(
    csv_path: str,
    *,
    timestamp_column: str | None = None,
    ticker_column: str | None = None,
):
    """Load and normalize market CSV data for ticker-wise engine processing."""
    if pd is None:
        return _load_market_data_stdlib(
            csv_path,
            timestamp_column=timestamp_column,
            ticker_column=ticker_column,
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    resolved_timestamp = _resolve_column(df.columns, TIMESTAMP_ALIASES, timestamp_column)
    resolved_ticker = _resolve_column(df.columns, TICKER_ALIASES, ticker_column)

    df = df.rename(columns={resolved_timestamp: "timestamp", resolved_ticker: "ticker"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df = df[(df["ticker"] != "") & (df["ticker"].str.lower() != "nan")]
    df = df.dropna(subset=["timestamp"]).copy()

    for col in df.columns:
        if col in {"timestamp", "ticker"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "value" not in df.columns:
        close_col = _resolve_optional_column(df.columns, CLOSE_ALIASES)
        if close_col is not None:
            df["value"] = df[close_col]

    df = df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return df


__all__ = ["load_market_data", "TIMESTAMP_ALIASES", "TICKER_ALIASES", "CLOSE_ALIASES"]
