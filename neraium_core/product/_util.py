"""Shared helpers for deterministic product-layer builders."""

from __future__ import annotations

from typing import Any


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def safe_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = mapping
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur
