from __future__ import annotations

from typing import Any

import numpy as np


def vector_from_frame(
    frame: dict[str, Any],
    *,
    sensor_order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build a stable numeric vector from frame sensor values.

    Returns:
      - vector: float array aligned to `resolved_sensor_order`
      - resolved_sensor_order: existing order or inferred sorted order
    """
    sensor_values = frame["sensor_values"]
    resolved_sensor_order = list(sensor_order or sorted(sensor_values.keys()))

    values: list[float] = []
    for name in resolved_sensor_order:
        v = sensor_values.get(name)
        try:
            values.append(float(v) if v is not None else float("nan"))
        except (TypeError, ValueError):
            values.append(float("nan"))

    return np.asarray(values, dtype=float), resolved_sensor_order


def window_from_frames(
    frames_list: list[dict[str, Any]],
    *,
    size: int,
    stride: int = 1,
    recent: bool,
) -> np.ndarray | None:
    if len(frames_list) < size:
        return None

    selected = frames_list[-size:] if recent else frames_list[:size]
    vectors = np.stack([f["_vector"] for f in selected], axis=0)
    vectors = vectors[:: max(1, int(stride))]
    if vectors.shape[0] < 2:
        return None
    return vectors


def timestamps_from_frames(
    frames_list: list[dict[str, Any]],
    *,
    size: int,
    recent: bool,
) -> list[float] | None:
    if len(frames_list) < size:
        return None

    selected = frames_list[-size:] if recent else frames_list[:size]
    ts_vals: list[float] = []
    for f in selected:
        try:
            ts_vals.append(float(f.get("timestamp")))
        except (TypeError, ValueError):
            continue

    return ts_vals if len(ts_vals) >= 2 else None
