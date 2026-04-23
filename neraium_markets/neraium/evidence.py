"""Day 10: append-safe JSONL evidence logging for auditability."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config import DATE_COLUMN

_DEFAULT_PATH = Path("output") / "evidence_log.jsonl"


def _json_default(o: object) -> str:
    if isinstance(o, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(o).isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _get(row: Mapping[str, Any] | pd.Series, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(row, pd.Series):
            if k in row.index:
                v = row[k]
                if pd.isna(v) and k != keys[-1]:
                    continue
                return v
        elif k in row:
            return row[k]
    return default


def _evidence_payload(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Build one JSON-serializable evidence record."""
    ts = _get(row, DATE_COLUMN, "timestamp", default=None)
    if ts is not None and not isinstance(ts, str):
        ts = pd.Timestamp(ts).isoformat()

    expl = _get(row, "path_aware_market_explanation", "market_explanation", "explanation", default="")
    if expl is None or (isinstance(expl, float) and pd.isna(expl)):
        expl = ""

    return {
        "timestamp": ts,
        "regime_label": str(_get(row, "market_regime_label", "regime_label", default="")),
        "confidence_score": float(_get(row, "market_confidence_score", "confidence_score", default=float("nan"))),
        "action_posture": str(_get(row, "market_action_posture", "action_posture", default="")),
        "path_adjusted_market_action": str(_get(row, "path_adjusted_market_action", default="")),
        "warning_level": str(_get(row, "market_warning_level", "warning_level", default="")),
        "scenario_path_label": str(_get(row, "scenario_path_label", default="")),
        "trajectory_direction": str(_get(row, "trajectory_direction", default="")),
        "instability_score": float(_get(row, "market_instability_score", "instability_score", default=float("nan"))),
        "coherence_score": float(_get(row, "market_coherence_score", "coherence_score", default=float("nan"))),
        "explanation": str(expl),
    }


def log_signal_event(
    row: Mapping[str, Any] | pd.Series,
    path: str | Path = _DEFAULT_PATH,
) -> None:
    """Append one evidence record as a JSON line (append-safe)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _evidence_payload(row)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n")


def log_batch_events(df: pd.DataFrame, path: str | Path = _DEFAULT_PATH) -> None:
    """Append one JSON line per row of ``df`` (sorted by timestamp when present)."""
    if df.empty:
        return
    work = df.copy()
    if DATE_COLUMN in work.columns:
        work = work.sort_values(DATE_COLUMN, ascending=True)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for _, row in work.iterrows():
            payload = _evidence_payload(row)
            fh.write(json.dumps(payload, default=_json_default, ensure_ascii=False) + "\n")
