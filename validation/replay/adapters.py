from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _normalize_timestamp(value: Any, fallback_index: int) -> float:
    if value is None:
        return float(fallback_index)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback_index)


def _normalize_record(raw: dict[str, Any], idx: int) -> dict[str, Any]:
    record = dict(raw)
    record["timestamp"] = _normalize_timestamp(record.get("timestamp"), idx)
    record.setdefault("asset_id", str(record.get("asset") or record.get("unit") or "unknown"))
    record.setdefault("observation", {})
    return record


def load_csv_telemetry(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            observation = {
                k: (float(v) if isinstance(v, str) and v.strip() and v.replace(".", "", 1).replace("-", "", 1).isdigit() else v)
                for k, v in row.items()
                if k not in {"timestamp", "asset_id", "actual_intervention", "outcome_label", "outcome_score"}
            }
            rows.append(
                _normalize_record(
                    {
                        **row,
                        "observation": observation,
                        "actual_intervention": row.get("actual_intervention") or None,
                        "outcome_label": row.get("outcome_label") or None,
                        "outcome_score": float(row["outcome_score"]) if row.get("outcome_score") else None,
                    },
                    idx,
                )
            )
    return sorted(rows, key=lambda r: (float(r.get("timestamp", 0.0)), str(r.get("asset_id", ""))))


def load_json_logs(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("events", payload if isinstance(payload, list) else [])
    normalized = [_normalize_record(dict(row), idx) for idx, row in enumerate(rows)]
    return sorted(normalized, key=lambda r: (float(r.get("timestamp", 0.0)), str(r.get("asset_id", ""))))


def load_event_logs(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        record = {
            "timestamp": parts[0] if len(parts) > 0 else idx,
            "asset_id": parts[1] if len(parts) > 1 else "unknown",
            "observation": {"event": parts[2] if len(parts) > 2 else "unknown"},
            "actual_intervention": parts[3] if len(parts) > 3 and parts[3] else None,
            "outcome_label": parts[4] if len(parts) > 4 and parts[4] else None,
        }
        rows.append(_normalize_record(record, idx))
    return sorted(rows, key=lambda r: (float(r.get("timestamp", 0.0)), str(r.get("asset_id", ""))))


def load_dataset(path: str | Path, fmt: str) -> list[dict[str, Any]]:
    fmt_norm = fmt.lower().strip()
    if fmt_norm == "csv":
        return load_csv_telemetry(path)
    if fmt_norm == "json":
        return load_json_logs(path)
    if fmt_norm in {"event", "events", "log"}:
        return load_event_logs(path)
    raise ValueError(f"Unsupported dataset format: {fmt}")
