from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from neraium_core.grow.scopes import SCOPE_ORDER, scope_id_for_record


def records_by_scope(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {scope: defaultdict(list) for scope in SCOPE_ORDER}
    for row in records:
        for scope in SCOPE_ORDER:
            scope_id = scope_id_for_record(scope, row)
            if scope_id:
                grouped[scope][scope_id].append(row)
    return grouped


def aggregate_signals(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in (row.get("signals") or {}).items():
            if isinstance(value, (int, float)):
                buckets[key].append(float(value))
    return {key: round(mean(values), 4) for key, values in buckets.items()}


def latest_timestamp(rows: list[dict[str, Any]]) -> str:
    return str(sorted(rows, key=lambda row: row["timestamp"])[-1]["timestamp"])
