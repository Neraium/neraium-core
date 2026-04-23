"""Day 14: detect new data, trim windows, merge incremental outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def detect_new_data(
    normalized_data: dict[str, pd.DataFrame],
    prior_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare current normalized frames to ``prior_state['latest_timestamp_by_asset']``.

    ISO timestamps in prior state are compared to max ``timestamp`` per asset (pandas).
    """
    prior_ts: dict[str, Any] = prior_state.get("latest_timestamp_by_asset") or {}
    latest_by_asset: dict[str, str | None] = {}
    has_new: dict[str, bool] = {}
    newest_list: list[pd.Timestamp] = []
    oldest_new_list: list[pd.Timestamp] = []

    for asset, df in normalized_data.items():
        if df is None or df.empty or "timestamp" not in df.columns:
            latest_by_asset[asset] = None
            has_new[asset] = True
            continue
        cur_max = pd.Timestamp(df["timestamp"].max())
        latest_by_asset[asset] = cur_max.isoformat()
        prev_raw = prior_ts.get(asset)
        if prev_raw is None:
            has_new[asset] = True
            newest_list.append(cur_max)
            oldest_new_list.append(cur_max)
            continue
        prev = pd.Timestamp(prev_raw)
        if cur_max > prev:
            has_new[asset] = True
            sub = df[pd.to_datetime(df["timestamp"]) > prev]
            if not sub.empty:
                oldest_new_list.append(pd.Timestamp(sub["timestamp"].min()))
            newest_list.append(cur_max)
        else:
            has_new[asset] = False

    any_new = any(has_new.values())
    newest_ts = max(newest_list) if newest_list else None
    oldest_new_ts = min(oldest_new_list) if oldest_new_list else None

    return {
        "latest_timestamp_by_asset": latest_by_asset,
        "has_new_data_by_asset": has_new,
        "any_new_data": any_new,
        "oldest_new_timestamp": oldest_new_ts.isoformat() if oldest_new_ts is not None else None,
        "newest_new_timestamp": newest_ts.isoformat() if newest_ts is not None else None,
    }


def filter_to_incremental_window(
    normalized_data: dict[str, pd.DataFrame],
    prior_state: dict[str, Any],
    lookback_buffer: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    If no prior timestamps, return full ``normalized_data``.

    Otherwise keep rows with ``timestamp`` strictly after (last processed - buffer rows)
    per asset, but never return fewer than ``lookback_buffer`` rows when data exists
    (so rolling features still have context).
    """
    prior_ts: dict[str, Any] = prior_state.get("latest_timestamp_by_asset") or {}
    if not prior_ts:
        return {k: v.copy() for k, v in normalized_data.items()}

    out: dict[str, pd.DataFrame] = {}
    for asset, df in normalized_data.items():
        if df is None or df.empty:
            out[asset] = df
            continue
        prev_raw = prior_ts.get(asset)
        if prev_raw is None:
            out[asset] = df.copy()
            continue
        prev = pd.Timestamp(prev_raw)
        ts = pd.to_datetime(df["timestamp"])
        mask = ts > prev
        sub = df.loc[mask].copy()
        if len(sub) < lookback_buffer and len(df) >= lookback_buffer:
            sub = df.iloc[-lookback_buffer:].copy()
        elif sub.empty and len(df) > 0:
            sub = df.iloc[-lookback_buffer:].copy() if len(df) >= lookback_buffer else df.copy()
        out[asset] = sub.sort_values("timestamp").reset_index(drop=True)
    return out


def merge_incremental_outputs(
    prior_output: pd.DataFrame,
    new_output: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Concatenate, sort by ``timestamp``, drop duplicate timestamps keeping the new row."""
    if prior_output is None or prior_output.empty:
        return new_output.copy() if new_output is not None else pd.DataFrame()
    if new_output is None or new_output.empty:
        return prior_output.copy()
    comb = pd.concat([prior_output, new_output], ignore_index=True)
    if timestamp_col not in comb.columns:
        return comb
    comb[timestamp_col] = pd.to_datetime(comb[timestamp_col])
    comb = comb.sort_values(timestamp_col, ascending=True)
    comb = comb.drop_duplicates(subset=[timestamp_col], keep="last")
    return comb.reset_index(drop=True)


def build_resume_summary(
    *,
    prior_state_found: bool,
    incremental_mode_used: bool,
    new_data_summary: dict[str, Any] | None,
    cache_hits: int,
    cache_misses: int,
    input_snapshot_written: bool,
    output_snapshot_written: bool,
    state_updated: bool,
    skipped_full_pipeline: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    """Operator-facing summary for ``run_resume_summary.json``."""
    return {
        "prior_state_found": prior_state_found,
        "incremental_mode_used": incremental_mode_used,
        "new_data_detection": new_data_summary or {},
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "input_snapshot_written": input_snapshot_written,
        "output_snapshot_written": output_snapshot_written,
        "state_updated": state_updated,
        "skipped_full_pipeline": skipped_full_pipeline,
        "notes": notes,
    }
