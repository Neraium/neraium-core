from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neraium_core.features.window_feature_extractor import extract_window_features


@dataclass(frozen=True)
class RawTelemetrySlice:
    unit: str
    cycle: int
    timestamp: str | None
    values: np.ndarray
    channel_names: list[str]
    operating_context: dict[str, Any]


def _safe_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("raw telemetry slice must be 1D or 2D")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _load_json(path: Path) -> list[RawTelemetrySlice]:
    obj = json.loads(path.read_text())
    rows = obj if isinstance(obj, list) else obj.get("windows", [])
    out: list[RawTelemetrySlice] = []
    for idx, row in enumerate(rows):
        values = _safe_matrix(np.asarray(row.get("values", []), dtype=float))
        channel_names = row.get("channel_names") or [f"ch_{i}" for i in range(values.shape[1])]
        out.append(
            RawTelemetrySlice(
                unit=str(row.get("unit", path.stem)),
                cycle=int(row.get("cycle", idx + 1)),
                timestamp=row.get("timestamp"),
                values=values,
                channel_names=[str(v) for v in channel_names],
                operating_context=dict(row.get("operating_context", {})),
            )
        )
    return out


def _load_csv(path: Path) -> list[RawTelemetrySlice]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        channel_names = [h for h in (reader.fieldnames or []) if h not in {"unit", "cycle", "timestamp"}]
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for row in reader:
            unit = str(row.get("unit") or path.stem)
            cycle = int(row.get("cycle") or 1)
            key = (unit, cycle)
            rec = grouped.setdefault(
                key,
                {
                    "timestamp": row.get("timestamp"),
                    "rows": [],
                },
            )
            rec["rows"].append([float(row.get(c) or 0.0) for c in channel_names])
    out: list[RawTelemetrySlice] = []
    for (unit, cycle), payload in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        out.append(
            RawTelemetrySlice(
                unit=unit,
                cycle=cycle,
                timestamp=payload["timestamp"],
                values=_safe_matrix(np.asarray(payload["rows"], dtype=float)),
                channel_names=[str(v) for v in channel_names],
                operating_context={},
            )
        )
    return out


def _load_numpy(path: Path) -> list[RawTelemetrySlice]:
    if path.suffix == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
    else:
        npz = np.load(path, allow_pickle=True)
        arr = np.asarray(npz["windows"], dtype=float)
    out: list[RawTelemetrySlice] = []
    for i in range(arr.shape[0]):
        values = _safe_matrix(arr[i])
        out.append(
            RawTelemetrySlice(
                unit=path.stem,
                cycle=i + 1,
                timestamp=None,
                values=values,
                channel_names=[f"ch_{j}" for j in range(values.shape[1])],
                operating_context={},
            )
        )
    return out


def load_raw_telemetry_windows(input_path: str | Path) -> list[RawTelemetrySlice]:
    path = Path(input_path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in {".json", ".csv", ".npy", ".npz"}]
    else:
        raise FileNotFoundError(f"Input path not found: {path}")

    all_slices: list[RawTelemetrySlice] = []
    for f in files:
        ext = f.suffix.lower()
        if ext == ".json":
            all_slices.extend(_load_json(f))
        elif ext == ".csv":
            all_slices.extend(_load_csv(f))
        elif ext in {".npy", ".npz"}:
            all_slices.extend(_load_numpy(f))

    all_slices.sort(key=lambda s: (s.unit, s.cycle, str(s.timestamp or "")))
    return all_slices


def convert_raw_telemetry_to_structural_rows(input_path: str | Path) -> list[dict[str, Any]]:
    slices = load_raw_telemetry_windows(input_path)
    rows: list[dict[str, Any]] = []
    for s in slices:
        features = extract_window_features(s.values, channel_names=s.channel_names)
        row: dict[str, Any] = {
            "unit": s.unit,
            "cycle": int(s.cycle),
            "timestamp": s.timestamp or "",
            "operating_context": json.dumps(s.operating_context, sort_keys=True),
        }
        row.update({f"sensor__{k}": float(v) for k, v in features.flattened.items()})
        rows.append(row)
    return rows


def write_structural_csv(rows: list[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def raw_slices_to_structural_frames(
    slices: list[RawTelemetrySlice],
    *,
    site_id: str = "raw-telemetry",
    customer_id: str = "default-customer",
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for s in slices:
        features = extract_window_features(s.values, channel_names=s.channel_names)
        sensor_values = {f"sensor__{k}": float(v) for k, v in features.flattened.items()}
        frame: dict[str, Any] = {
            "timestamp": s.timestamp or str(s.cycle),
            "customer_id": customer_id,
            "site_id": site_id,
            "asset_id": s.unit,
            "sensor_values": sensor_values,
            "raw_window_metadata": {
                "unit": s.unit,
                "cycle": int(s.cycle),
                "operating_context": dict(s.operating_context),
            },
        }
        frames.append(frame)
    return frames
