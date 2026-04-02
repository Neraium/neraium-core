from __future__ import annotations

from typing import Any, Dict, Iterable, List


def normalize_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Normalize record values into float-only dictionaries."""
    normalized: List[Dict[str, float]] = []
    for row in records:
        normalized.append({str(k): float(v) for k, v in row.items()})
    return normalized
