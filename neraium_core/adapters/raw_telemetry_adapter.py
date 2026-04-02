from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neraium_core.features.window_feature_extractor import extract_window_features
from neraium_core.pipeline import build_frame

_SUPPORTED_EXTENSIONS = {".json", ".csv", ".npy", ".npz", ".txt"}

# First line looks like whitespace-separated numeric samples (IMS bearing chunks, etc.).
_NUMERIC_SIGNAL_FIRST_LINE = re.compile(
    r"^\s*[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?:[\s,;]+[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)*\s*$"
)

_METADATA_HINTS = {
    "timestamp",
    "time",
    "datetime",
    "date",
    "site",
    "site_id",
    "plant",
    "factory",
    "line",
    "asset",
    "asset_id",
    "machine",
    "unit",
    "unit_id",
    "equipment",
    "cycle",
    "step",
    "index",
    "seq",
    "id",
    "meta",
    "context",
    "batch",
    "lot",
    "run",
}


@dataclass(frozen=True)
class RawTelemetrySlice:
    unit: str
    cycle: int
    timestamp: str | None
    values: np.ndarray
    channel_names: list[str]
    operating_context: dict[str, Any]


@dataclass(frozen=True)
class IngestionDiagnostics:
    detected_input_type: str
    timestep_count: int
    sensor_count: int
    preprocessing_mode: str
    has_valid_signals: bool
    sensor_columns: list[str]
    metadata_columns: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class RawIndustrialIngestionResult:
    frames: list[dict[str, Any]]
    diagnostics: IngestionDiagnostics


@dataclass(frozen=True)
class SchemaMapping:
    sensor_columns: list[str]
    metadata_columns: list[str]
    timestamp_column: str | None
    asset_column: str | None
    site_column: str | None


def _safe_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("raw telemetry slice must be 1D or 2D")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_signal_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "signal"


def _is_probable_metadata_column(name: str) -> bool:
    norm = _normalize_signal_name(name).lower()
    return norm in _METADATA_HINTS or any(norm.startswith(f"{h}_") for h in _METADATA_HINTS)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _looks_like_numeric_token(value: str) -> bool:
    try:
        float(str(value).strip())
        return True
    except (TypeError, ValueError):
        return False


def _infer_table_schema(rows: list[dict[str, Any]], headers: list[str]) -> SchemaMapping:
    lower_map = {h: _normalize_signal_name(h).lower() for h in headers}

    ts_col = next(
        (
            h
            for h, l in lower_map.items()
            if l in {"timestamp", "time", "datetime", "date"}
            or l.endswith("_timestamp")
            or l.endswith("_time")
            or l.endswith("_at")
        ),
        None,
    )
    asset_col = next(
        (
            h
            for h, l in lower_map.items()
            if l in {"asset", "asset_id", "machine", "unit", "unit_id", "equipment"}
            or l.endswith("_asset")
            or l.endswith("_machine")
            or l.endswith("_unit")
            or l.endswith("_tag")
        ),
        None,
    )
    site_col = next((h for h, l in lower_map.items() if l in {"site", "site_id", "plant", "factory", "line"}), None)

    sensor_cols: list[str] = []
    metadata_cols: list[str] = []

    sample_rows = rows[: min(len(rows), 32)]
    for h in headers:
        if h in {ts_col, asset_col, site_col}:
            metadata_cols.append(h)
            continue
        if _is_probable_metadata_column(h):
            metadata_cols.append(h)
            continue
        numeric = 0
        for row in sample_rows:
            if _to_float(row.get(h)) is not None:
                numeric += 1
        if numeric > 0:
            sensor_cols.append(h)
        else:
            metadata_cols.append(h)

    return SchemaMapping(
        sensor_columns=sensor_cols,
        metadata_columns=metadata_cols,
        timestamp_column=ts_col,
        asset_column=asset_col,
        site_column=site_col,
    )


def _load_tabular_rows(path: Path) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".csv":
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(r) for r in reader if r is not None]
    if ext == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            return [dict(r) for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            if isinstance(obj.get("windows"), list):
                return []
            if isinstance(obj.get("rows"), list):
                return [dict(r) for r in obj["rows"] if isinstance(r, dict)]
            if isinstance(obj.get("data"), list):
                return [dict(r) for r in obj["data"] if isinstance(r, dict)]
            return [dict(obj)]
    return []


def _build_frames_from_tabular_rows(
    rows: list[dict[str, Any]],
    *,
    source_label: str,
    customer_id: str,
    default_site_id: str,
    default_asset_id: str,
) -> tuple[list[dict[str, Any]], SchemaMapping, list[str]]:
    if not rows:
        return [], SchemaMapping([], [], None, None, None), ["No rows found in tabular input."]

    headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    schema = _infer_table_schema(rows, headers)
    warnings: list[str] = []
    if not schema.sensor_columns:
        warnings.append("Schema discovery found no valid numeric signal columns.")

    frames: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        sensor_values = {_normalize_signal_name(c): _to_float(row.get(c)) for c in schema.sensor_columns}
        frame = build_frame(
            timestamp=row.get(schema.timestamp_column) if schema.timestamp_column else None,
            customer_id=customer_id,
            site_id=row.get(schema.site_column) if schema.site_column else default_site_id,
            asset_id=row.get(schema.asset_column) if schema.asset_column else default_asset_id,
            sensor_values=sensor_values,
        )
        frame["raw_window_metadata"] = {
            "source": source_label,
            "cycle": idx,
            "metadata": {k: row.get(k) for k in schema.metadata_columns},
            "schema": {
                "timestamp_column": schema.timestamp_column,
                "asset_column": schema.asset_column,
                "site_column": schema.site_column,
            },
        }
        frames.append(frame)

    return frames, schema, warnings


def _try_parse_file_as_tabular_row(path: Path) -> dict[str, Any] | None:
    ext = path.suffix.lower()
    try:
        if ext == ".json":
            obj = json.loads(path.read_text())
            if isinstance(obj, dict):
                return obj
        if ext == ".csv":
            with path.open("r", newline="") as f:
                reader = csv.DictReader(f)
                row = next(iter(reader), None)
                if row is None:
                    return None
                keys = [str(k) for k in row.keys()]
                if keys and all(_looks_like_numeric_token(k) for k in keys):
                    return None
                return row
    except Exception:
        return None
    return None


def _load_directory_rows(path: Path) -> list[dict[str, Any]]:
    files = [p for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in {".json", ".csv"}]
    rows: list[dict[str, Any]] = []
    for file_path in files:
        row = _try_parse_file_as_tabular_row(file_path)
        if row:
            rows.append(row)
    return rows


def _collect_nested_signal_files(root: Path) -> list[Path]:
    """Files under ``root`` (recursive): known extensions + extensionless IMS-style numeric blocks."""
    found: set[Path] = set()
    for ext in _SUPPORTED_EXTENSIONS:
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            found.update(root.rglob(pattern))
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix:
            continue
        if candidate.name.startswith("."):
            continue
        try:
            first_line = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        except OSError:
            continue
        if first_line and _NUMERIC_SIGNAL_FIRST_LINE.match(first_line[0].strip()):
            found.add(candidate)
    return sorted(found)


def _load_directory_slices(path: Path) -> tuple[list[RawTelemetrySlice], list[str]]:
    files = _collect_nested_signal_files(path)
    warnings: list[str] = []
    out: list[RawTelemetrySlice] = []
    cycle = 1
    for f in files:
        ext = f.suffix.lower()
        try:
            if ext in {".npy", ".npz"}:
                loaded = np.load(f, allow_pickle=True)
                arr = np.asarray(loaded["windows"], dtype=float) if ext == ".npz" else np.asarray(loaded, dtype=float)
            elif ext == ".json":
                obj = json.loads(f.read_text())
                payload = obj.get("values") if isinstance(obj, dict) and "values" in obj else obj
                arr = np.asarray(payload, dtype=float)
            elif ext == ".csv":
                arr = np.loadtxt(f, dtype=float, delimiter=",")
            else:
                # .txt and extensionless IMS files: whitespace-separated numeric rows
                arr = np.loadtxt(f, dtype=float)
            arr = _safe_matrix(arr)
            try:
                rel = f.relative_to(path)
                unit_key = str(rel.as_posix())
            except ValueError:
                unit_key = f.name
            out.append(
                RawTelemetrySlice(
                    unit=unit_key,
                    cycle=cycle,
                    timestamp=None,
                    values=arr,
                    channel_names=[f"ch_{i}" for i in range(arr.shape[1])],
                    operating_context={"source_file": f.name, "relative_path": unit_key},
                )
            )
            cycle += 1
        except Exception as exc:
            warnings.append(f"Skipped {f.name}: {exc}")
    return out, warnings


def load_raw_telemetry_windows(input_path: str | Path) -> list[RawTelemetrySlice]:
    path = Path(input_path)
    if path.is_dir():
        slices, _warnings = _load_directory_slices(path)
        return slices
    ext = path.suffix.lower()
    if ext == ".json":
        obj = json.loads(path.read_text())
        rows = obj if isinstance(obj, list) else obj.get("windows", [])
        out: list[RawTelemetrySlice] = []
        for idx, row in enumerate(rows):
            values = _safe_matrix(np.asarray(row.get("values", []), dtype=float))
            out.append(
                RawTelemetrySlice(
                    unit=str(row.get("unit", path.stem)),
                    cycle=int(row.get("cycle", idx + 1)),
                    timestamp=row.get("timestamp"),
                    values=values,
                    channel_names=[str(v) for v in (row.get("channel_names") or [f"ch_{i}" for i in range(values.shape[1])])],
                    operating_context=dict(row.get("operating_context", {})),
                )
            )
        return out
    if ext == ".csv":
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            channel_names = [h for h in (reader.fieldnames or []) if h not in {"unit", "cycle", "timestamp"}]
            grouped: dict[tuple[str, int], dict[str, Any]] = {}
            for row in reader:
                unit = str(row.get("unit") or path.stem)
                cycle = int(row.get("cycle") or 1)
                rec = grouped.setdefault((unit, cycle), {"timestamp": row.get("timestamp"), "rows": []})
                rec["rows"].append([float(row.get(c) or 0.0) for c in channel_names])
        out: list[RawTelemetrySlice] = []
        for (unit, cyc), payload in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
            out.append(
                RawTelemetrySlice(
                    unit=unit,
                    cycle=cyc,
                    timestamp=payload["timestamp"],
                    values=_safe_matrix(np.asarray(payload["rows"], dtype=float)),
                    channel_names=[str(v) for v in channel_names],
                    operating_context={},
                )
            )
        return out
    if ext in {".npy", ".npz"}:
        arr = np.asarray(np.load(path, allow_pickle=True), dtype=float) if ext == ".npy" else np.asarray(np.load(path, allow_pickle=True)["windows"], dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        out = []
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
    raise FileNotFoundError(f"Unsupported input path: {path}")


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
        frame = build_frame(
            timestamp=s.timestamp or None,
            customer_id=customer_id,
            site_id=site_id,
            asset_id=s.unit,
            sensor_values=sensor_values,
        )
        frame["raw_window_metadata"] = {
            "unit": s.unit,
            "cycle": int(s.cycle),
            "operating_context": dict(s.operating_context),
        }
        frames.append(frame)
    return frames


def ingest_raw_industrial_input(
    input_path: str | Path,
    *,
    customer_id: str = "default-customer",
    site_id: str = "raw-telemetry",
    default_asset_id: str = "raw-asset",
    preprocessing_mode: str = "auto",
    sample_size: int | None = None,
) -> RawIndustrialIngestionResult:
    path = Path(input_path)
    warnings: list[str] = []

    if preprocessing_mode not in {"auto", "none", "waveform_features"}:
        raise ValueError("preprocessing_mode must be one of: auto, none, waveform_features")

    rows = _load_tabular_rows(path) if path.is_file() else _load_directory_rows(path) if path.is_dir() else []
    detected_input_type = "tabular_timeseries" if rows else "directory_signal_blocks"

    if rows:
        frames, schema, schema_warnings = _build_frames_from_tabular_rows(
            rows,
            source_label=str(path),
            customer_id=customer_id,
            default_site_id=site_id,
            default_asset_id=default_asset_id,
        )
        warnings.extend(schema_warnings)
        sensor_cols = list(schema.sensor_columns)
        metadata_cols = list(schema.metadata_columns)
        effective_preprocessing = "none" if preprocessing_mode in {"auto", "none"} else preprocessing_mode
    else:
        if path.is_dir():
            slices, slice_warnings = _load_directory_slices(path)
            warnings.extend(slice_warnings)
        else:
            slices = load_raw_telemetry_windows(path)
        frames = raw_slices_to_structural_frames(slices, site_id=site_id, customer_id=customer_id)
        sensor_cols = sorted({k for frame in frames for k in frame.get("sensor_values", {}).keys()})
        metadata_cols = ["unit", "cycle", "operating_context"]
        effective_preprocessing = "waveform_features" if preprocessing_mode == "auto" else preprocessing_mode

    if sample_size is not None and sample_size > 0:
        frames = frames[:sample_size]
        warnings.append(f"Validation mode enabled: limited to first {len(frames)} timesteps.")

    has_valid_signals = any(any(v is not None for v in frame.get("sensor_values", {}).values()) for frame in frames)
    if not has_valid_signals:
        warnings.append("No valid numeric signals detected after preprocessing.")

    diagnostics = IngestionDiagnostics(
        detected_input_type=detected_input_type,
        timestep_count=len(frames),
        sensor_count=len(sensor_cols),
        preprocessing_mode=effective_preprocessing,
        has_valid_signals=has_valid_signals,
        sensor_columns=list(sensor_cols),
        metadata_columns=list(metadata_cols),
        warnings=warnings,
    )
    return RawIndustrialIngestionResult(frames=frames, diagnostics=diagnostics)


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
