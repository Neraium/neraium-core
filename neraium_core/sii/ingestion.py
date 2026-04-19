from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from io import StringIO
import math
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SIIValidationError
from .types import (
    CanonicalAnalysisTable,
    CanonicalAnalysisWindow,
    CanonicalIngestionRecord,
    IngestionQualityMetadata,
    IngestionSourceMetadata,
    TelemetryFrame,
    ingestion_record_to_frame,
)


REQUIRED_CSV_COLUMNS = {"timestamp", "site_id", "asset_id"}
_RECOGNIZED_COLUMNS = {
    "timestamp",
    "site_id",
    "system_id",
    "asset_id",
    "node_id",
    "quality_metadata",
    "source_metadata",
    "metadata",
}

DEFAULT_SITE_ID = "default-site"
DEFAULT_SYSTEM_ID = "default-system"
DEFAULT_ASSET_ID = "default-asset"
DEFAULT_NODE_ID = "default-node"


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            fv = float(text)
            if not (fv == fv) or abs(fv) == float("inf"):
                return None
            return fv
        except ValueError as exc:
            raise SIIValidationError(f"Invalid numeric value: {value!r}") from exc
    if isinstance(value, (int, float)):
        fv = float(value)
        if not (fv == fv) or abs(fv) == float("inf"):
            return None
        return fv
    raise SIIValidationError(f"Unsupported value type: {type(value).__name__}")


def _normalize_timestamp_to_epoch_seconds(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        fv = float(text)
        if fv == fv and abs(fv) != float("inf"):
            return str(fv)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return str(dt.timestamp())


def _as_str_or_default(value: Any, *, default: str) -> str:
    text = str(value if value is not None else "").strip()
    return text or default


def _parse_variables_map(
    payload: dict[str, Any],
) -> tuple[dict[str, float | None], list[str]]:
    raw_variables = payload.get("variables")
    if raw_variables is None:
        raw_variables = payload.get("sensor_values")
    if not isinstance(raw_variables, dict):
        raise SIIValidationError("variables (or sensor_values) must be an object")

    parsed: dict[str, float | None] = {}
    issues: list[str] = []
    for raw_name, raw_value in raw_variables.items():
        name = str(raw_name).strip()
        if not name:
            raise SIIValidationError("Variable name cannot be empty")
        try:
            parsed[name] = _as_float_or_none(raw_value)
        except SIIValidationError as exc:
            parsed[name] = None
            issues.append(f"invalid variable value for {name!r}: {exc}")
    return parsed, issues


def _default_flatlined_variables(variables: dict[str, float | None]) -> list[str]:
    finite = {k: float(v) for k, v in variables.items() if v is not None}
    if len(finite) < 2:
        return []
    values = list(finite.values())
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-12):
        return sorted(finite.keys())
    return []


def _build_quality_metadata(
    *,
    variables: dict[str, float | None],
    raw_quality: Any,
    validation_issues: list[str],
) -> IngestionQualityMetadata:
    total = max(1, len(variables))
    missing = sum(1 for v in variables.values() if v is None)
    default_missingness = float(missing) / float(total)
    default_completeness = float(max(0.0, min(1.0, 1.0 - default_missingness)))
    default_flatlined = _default_flatlined_variables(variables)
    default_stale = bool(default_flatlined)
    default_status = "ok" if variables else "empty"

    if not isinstance(raw_quality, dict):
        return IngestionQualityMetadata(
            missingness_rate=default_missingness,
            stale_signal=default_stale,
            flatlined_variables=default_flatlined,
            source_status=default_status,
            validation_issues=list(validation_issues),
            completeness_score=default_completeness,
        )

    def _float_with_fallback(key: str, fallback: float) -> float:
        raw = raw_quality.get(key, fallback)
        try:
            fv = float(raw)
        except Exception:
            validation_issues.append(f"quality_metadata.{key} is invalid")
            return fallback
        return float(max(0.0, min(1.0, fv)))

    stale_signal = bool(raw_quality.get("stale_signal", default_stale))
    flatlined_raw = raw_quality.get("flatlined_variables", default_flatlined)
    if isinstance(flatlined_raw, list):
        flatlined_variables = [str(x) for x in flatlined_raw if str(x).strip()]
    else:
        flatlined_variables = default_flatlined
        validation_issues.append("quality_metadata.flatlined_variables is invalid")

    raw_issues = raw_quality.get("validation_issues", [])
    issues_from_payload = [str(x) for x in raw_issues] if isinstance(raw_issues, list) else []
    source_status = _as_str_or_default(raw_quality.get("source_status"), default=default_status)

    return IngestionQualityMetadata(
        missingness_rate=_float_with_fallback("missingness_rate", default_missingness),
        stale_signal=stale_signal,
        flatlined_variables=flatlined_variables,
        source_status=source_status,
        validation_issues=list(dict.fromkeys([*validation_issues, *issues_from_payload])),
        completeness_score=_float_with_fallback("completeness_score", default_completeness),
    )


def _build_source_metadata(
    *,
    raw_source: Any,
    source_type: str,
    source_name: str,
) -> IngestionSourceMetadata:
    if not isinstance(raw_source, dict):
        return IngestionSourceMetadata(
            source_type=source_type,
            source_name=source_name,
        )
    tags_raw = raw_source.get("tags", {})
    tags = {str(k): str(v) for k, v in tags_raw.items()} if isinstance(tags_raw, dict) else {}
    return IngestionSourceMetadata(
        source_type=_as_str_or_default(raw_source.get("source_type"), default=source_type),
        source_name=_as_str_or_default(raw_source.get("source_name"), default=source_name),
        source_record_id=str(raw_source.get("source_record_id")).strip() or None
        if raw_source.get("source_record_id") is not None
        else None,
        source_schema_version=str(raw_source.get("source_schema_version")).strip() or None
        if raw_source.get("source_schema_version") is not None
        else None,
        request_id=str(raw_source.get("request_id")).strip() or None
        if raw_source.get("request_id") is not None
        else None,
        tags=tags,
    )


def ingestion_record_from_payload(
    payload: dict[str, Any],
    *,
    source_type: str = "payload",
    source_name: str = "payload",
) -> CanonicalIngestionRecord:
    if not isinstance(payload, dict):
        raise SIIValidationError("Payload must be an object")

    timestamp_issues: list[str] = []
    normalized_timestamp = _normalize_timestamp_to_epoch_seconds(payload.get("timestamp"))
    if normalized_timestamp is None:
        # Hardening behavior for pilot ingest: keep the row processable while
        # explicitly flagging the fallback in quality metadata for downstream
        # observability and triage.
        normalized_timestamp = "0"
        timestamp_issues.append("invalid_or_missing_timestamp_defaulted_to_epoch_zero")
    timestamp = normalized_timestamp
    site_id = _as_str_or_default(payload.get("site_id"), default=DEFAULT_SITE_ID)
    system_id = _as_str_or_default(payload.get("system_id"), default=DEFAULT_SYSTEM_ID)
    asset_id = _as_str_or_default(payload.get("asset_id"), default=DEFAULT_ASSET_ID)
    node_id = _as_str_or_default(payload.get("node_id"), default=DEFAULT_NODE_ID)

    variables, variable_issues = _parse_variables_map(payload)
    quality_metadata = _build_quality_metadata(
        variables=variables,
        raw_quality=payload.get("quality_metadata"),
        validation_issues=[*timestamp_issues, *variable_issues],
    )
    source_metadata = _build_source_metadata(
        raw_source=payload.get("source_metadata"),
        source_type=source_type,
        source_name=source_name,
    )
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise SIIValidationError("metadata must be an object when provided")

    return CanonicalIngestionRecord(
        timestamp=timestamp,
        site_id=site_id,
        system_id=system_id,
        asset_id=asset_id,
        node_id=node_id,
        variables=variables,
        quality_metadata=quality_metadata,
        source_metadata=source_metadata,
        metadata={str(k): v for k, v in metadata.items()},
    )


def ingestion_record_to_payload(record: CanonicalIngestionRecord) -> dict[str, Any]:
    return {
        "timestamp": record.timestamp,
        "site_id": record.site_id,
        "system_id": record.system_id,
        "asset_id": record.asset_id,
        "node_id": record.node_id,
        "variables": dict(record.variables),
        # Keep compatibility with legacy call sites while canonicalizing shape.
        "sensor_values": dict(record.variables),
        "quality_metadata": {
            "missingness_rate": record.quality_metadata.missingness_rate,
            "stale_signal": record.quality_metadata.stale_signal,
            "flatlined_variables": list(record.quality_metadata.flatlined_variables),
            "source_status": record.quality_metadata.source_status,
            "validation_issues": list(record.quality_metadata.validation_issues),
            "completeness_score": record.quality_metadata.completeness_score,
        },
        "source_metadata": {
            "source_type": record.source_metadata.source_type,
            "source_name": record.source_metadata.source_name,
            "source_record_id": record.source_metadata.source_record_id,
            "source_schema_version": record.source_metadata.source_schema_version,
            "request_id": record.source_metadata.request_id,
            "tags": dict(record.source_metadata.tags),
        },
        "metadata": dict(record.metadata),
    }


def frame_from_payload(payload: dict[str, Any]) -> TelemetryFrame:
    record = ingestion_record_from_payload(payload)
    return ingestion_record_to_frame(record)


def canonical_records_from_payloads(
    payloads: list[dict[str, Any]],
    *,
    source_type: str = "payload",
    source_name: str = "payload_batch",
) -> list[CanonicalIngestionRecord]:
    out: list[CanonicalIngestionRecord] = []
    for idx, payload in enumerate(payloads):
        try:
            out.append(
                ingestion_record_from_payload(
                    payload,
                    source_type=source_type,
                    source_name=source_name,
                )
            )
        except SIIValidationError as exc:
            raise SIIValidationError(f"Invalid payload at index {idx}: {exc}") from exc
    return out


def _fallback_ingest_code(*, source_type: str) -> str:
    if source_type == "csv":
        return "invalid_csv_row"
    if source_type == "json":
        return "invalid_json_item"
    return "invalid_payload"


def _classify_ingest_error_code(
    *,
    message: str,
    source_type: str,
    payload: Any | None = None,
) -> str:
    msg = message.lower()

    if "payload must be an object" in msg:
        return "non_object_payload"
    if "variables (or sensor_values) must be an object" in msg:
        if isinstance(payload, dict) and payload.get("variables") is None and payload.get("sensor_values") is None:
            return "missing_required_field"
        return _fallback_ingest_code(source_type=source_type)
    if "metadata must be an object" in msg:
        return "invalid_metadata_json"
    if "invalid numeric value" in msg or "unsupported value type" in msg:
        return "invalid_numeric_value"

    return _fallback_ingest_code(source_type=source_type)


def canonical_records_from_payloads_isolated(
    payloads: list[Any],
    *,
    source_type: str = "payload",
    source_name: str = "payload_batch",
) -> tuple[list[CanonicalIngestionRecord], list[dict[str, Any]]]:
    records: list[CanonicalIngestionRecord] = []
    ingest_errors: list[dict[str, Any]] = []
    for idx, payload in enumerate(payloads):
        try:
            if not isinstance(payload, dict):
                raise SIIValidationError("Payload must be an object")
            records.append(
                ingestion_record_from_payload(
                    payload,
                    source_type=source_type,
                    source_name=source_name,
                )
            )
        except SIIValidationError as exc:
            error: dict[str, Any] = {
                "index": idx,
                "code": _classify_ingest_error_code(
                    message=str(exc),
                    source_type=source_type,
                    payload=payload,
                ),
                "message": str(exc),
            }
            if isinstance(payload, dict):
                source_meta = payload.get("source_metadata")
                if isinstance(source_meta, dict) and source_meta.get("source_record_id") is not None:
                    error["source_record_id"] = str(source_meta.get("source_record_id"))
            ingest_errors.append(error)
    return records, ingest_errors


def frames_from_records(records: list[CanonicalIngestionRecord]) -> list[TelemetryFrame]:
    return [ingestion_record_to_frame(r) for r in records]


def _build_analysis_table(
    records: list[CanonicalIngestionRecord],
    *,
    variable_names: list[str] | None = None,
) -> CanonicalAnalysisTable:
    ordered_records = list(records)
    if variable_names is None:
        observed: set[str] = set()
        for record in ordered_records:
            observed.update(record.variables.keys())
        variable_names = sorted(observed)

    n_rows = len(ordered_records)
    n_cols = len(variable_names)
    values = np.full((n_rows, n_cols), np.nan, dtype=float)
    quality_mask = np.zeros((n_rows, n_cols), dtype=bool)
    epoch_timestamps = np.full((n_rows,), np.nan, dtype=float)
    site_ids: list[str] = []
    system_ids: list[str] = []
    asset_ids: list[str] = []
    node_ids: list[str] = []

    var_index = {name: i for i, name in enumerate(variable_names)}
    for row_idx, record in enumerate(ordered_records):
        try:
            epoch_timestamps[row_idx] = float(record.timestamp)
        except Exception:
            epoch_timestamps[row_idx] = np.nan
        site_ids.append(record.site_id)
        system_ids.append(record.system_id)
        asset_ids.append(record.asset_id)
        node_ids.append(record.node_id)
        for var_name, raw_value in record.variables.items():
            col_idx = var_index.get(var_name)
            if col_idx is None:
                continue
            if raw_value is None:
                continue
            fv = float(raw_value)
            if not (fv == fv) or abs(fv) == float("inf"):
                continue
            values[row_idx, col_idx] = fv
            quality_mask[row_idx, col_idx] = True

    return CanonicalAnalysisTable(
        records=ordered_records,
        variable_names=list(variable_names),
        epoch_timestamps=epoch_timestamps,
        values=values,
        quality_mask=quality_mask,
        site_ids=site_ids,
        system_ids=system_ids,
        asset_ids=asset_ids,
        node_ids=node_ids,
    )


def build_canonical_analysis_window(
    *,
    records: list[CanonicalIngestionRecord],
    baseline_window: int,
    recent_window: int,
    focal_site_id: str | None = None,
    focal_system_id: str | None = None,
    focal_asset_id: str | None = None,
    focal_node_id: str | None = None,
    variable_names: list[str] | None = None,
) -> CanonicalAnalysisWindow | None:
    if baseline_window < 1:
        raise SIIValidationError("baseline_window must be >= 1")
    if recent_window < 1:
        raise SIIValidationError("recent_window must be >= 1")
    if not records:
        return None

    selected = [
        r
        for r in records
        if (focal_site_id is None or r.site_id == focal_site_id)
        and (focal_system_id is None or r.system_id == focal_system_id)
        and (focal_asset_id is None or r.asset_id == focal_asset_id)
        and (focal_node_id is None or r.node_id == focal_node_id)
    ]
    if not selected:
        return None

    def _sort_key(record: CanonicalIngestionRecord) -> tuple[float, str, str, str, str]:
        normalized = _normalize_timestamp_to_epoch_seconds(record.timestamp)
        if normalized is None:
            ts = float("inf")
        else:
            try:
                ts = float(normalized)
            except Exception:
                ts = float("inf")
        return (ts, record.site_id, record.system_id, record.asset_id, record.node_id)

    raw_timestamps: list[float] = []
    for rec in selected:
        normalized = _normalize_timestamp_to_epoch_seconds(rec.timestamp)
        if normalized is None:
            continue
        try:
            raw_timestamps.append(float(normalized))
        except Exception:
            continue
    reordered_timestamps_detected = any(
        b < a for a, b in zip(raw_timestamps, raw_timestamps[1:])
    ) if len(raw_timestamps) >= 2 else False
    duplicate_timestamp_count = int(
        sum(1 for a, b in zip(raw_timestamps, raw_timestamps[1:]) if abs(b - a) < 1e-9)
    ) if len(raw_timestamps) >= 2 else 0
    invalid_timestamp_count = int(len(selected) - len(raw_timestamps))

    selected.sort(key=_sort_key)
    if len(selected) < recent_window:
        return None

    recent_records = selected[-recent_window:]
    if len(selected) >= (baseline_window + recent_window):
        baseline_records = selected[-(baseline_window + recent_window) : -recent_window]
    elif len(selected) >= baseline_window:
        baseline_records = selected[:baseline_window]
    else:
        baseline_records = selected[:recent_window]

    if variable_names is None:
        observed: set[str] = set()
        for rec in [*baseline_records, *recent_records]:
            observed.update(rec.variables.keys())
        variable_names = sorted(observed)

    baseline_table = _build_analysis_table(baseline_records, variable_names=variable_names)
    recent_table = _build_analysis_table(recent_records, variable_names=variable_names)
    quality_missingness = [
        float(r.quality_metadata.missingness_rate) for r in [*baseline_records, *recent_records]
    ]
    quality_summary = {
        "record_count": len(selected),
        "baseline_count": len(baseline_records),
        "recent_count": len(recent_records),
        "avg_missingness_rate": float(np.mean(quality_missingness)) if quality_missingness else 1.0,
        "stale_record_count": int(sum(1 for r in selected if r.quality_metadata.stale_signal)),
        "validation_issue_count": int(
            sum(len(r.quality_metadata.validation_issues) for r in selected)
        ),
        "source_statuses": sorted({r.quality_metadata.source_status for r in selected}),
        "invalid_timestamp_count": invalid_timestamp_count,
        "reordered_timestamps_detected": bool(reordered_timestamps_detected),
        "duplicate_timestamp_count": duplicate_timestamp_count,
    }
    focal = selected[-1]
    return CanonicalAnalysisWindow(
        baseline_table=baseline_table,
        recent_table=recent_table,
        baseline_matrix=np.asarray(baseline_table.values, dtype=float),
        recent_matrix=np.asarray(recent_table.values, dtype=float),
        variable_names=list(variable_names),
        focal_site_id=focal.site_id,
        focal_system_id=focal.system_id,
        focal_asset_id=focal.asset_id,
        focal_node_id=focal.node_id,
        time_range_start=baseline_records[0].timestamp if baseline_records else None,
        time_range_end=recent_records[-1].timestamp if recent_records else None,
        quality_summary=quality_summary,
    )


def _records_from_csv_text_isolated(
    csv_text: str,
    *,
    source_name: str,
) -> tuple[list[CanonicalIngestionRecord], list[dict[str, Any]]]:
    if not isinstance(csv_text, str):
        raise SIIValidationError("csv_text must be a string")

    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        return [], []

    headers = {h.strip() for h in reader.fieldnames if h is not None}
    if not REQUIRED_CSV_COLUMNS.issubset(headers):
        missing = sorted(REQUIRED_CSV_COLUMNS - headers)
        raise SIIValidationError(f"CSV missing required columns: {missing}")

    header_lookup: dict[str, str] = {}
    for h in reader.fieldnames:
        if h is None:
            continue
        norm = h.strip()
        if norm and norm not in header_lookup:
            header_lookup[norm] = h

    sensor_columns = [h for h in header_lookup.keys() if h not in _RECOGNIZED_COLUMNS]

    row_payloads: list[dict[str, Any]] = []
    pre_ingest_errors: list[dict[str, Any]] = []
    for row_index, row in enumerate(reader, start=2):
        def _parse_json_object_column(column_name: str) -> dict[str, Any] | None:
            raw_value = row.get(header_lookup.get(column_name, column_name))
            text = str(raw_value or "").strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise SIIValidationError(f"{column_name} must be valid JSON object") from exc
            if not isinstance(parsed, dict):
                raise SIIValidationError(f"{column_name} must be a JSON object")
            return parsed

        def _code_for_csv_row_error(message: str) -> str:
            msg = message.lower()
            if "quality_metadata" in msg:
                return "invalid_quality_metadata_json"
            if "source_metadata" in msg:
                return "invalid_source_metadata_json"
            if "metadata" in msg:
                return "invalid_metadata_json"
            return "invalid_csv_row"

        try:
            variables = {col: row.get(header_lookup[col]) for col in sensor_columns}
            payload = {
                "timestamp": row.get(header_lookup.get("timestamp", "timestamp")),
                "site_id": row.get(header_lookup.get("site_id", "site_id")),
                "system_id": row.get(header_lookup.get("system_id", "system_id")),
                "asset_id": row.get(header_lookup.get("asset_id", "asset_id")),
                "node_id": row.get(header_lookup.get("node_id", "node_id")),
                "variables": variables,
                "source_metadata": {
                    "source_type": "csv",
                    "source_name": "csv_file",
                    "source_record_id": str(row_index),
                },
            }
            quality_metadata = _parse_json_object_column("quality_metadata")
            if quality_metadata is not None:
                payload["quality_metadata"] = quality_metadata
            source_metadata = _parse_json_object_column("source_metadata")
            if source_metadata is not None:
                payload["source_metadata"] = source_metadata
                payload["source_metadata"]["source_record_id"] = payload["source_metadata"].get(
                    "source_record_id", str(row_index)
                )
            metadata = _parse_json_object_column("metadata")
            if metadata is not None:
                payload["metadata"] = metadata
            row_payloads.append(payload)
        except SIIValidationError as exc:
            pre_ingest_errors.append(
                {
                    "index": max(0, row_index - 2),
                    "code": _code_for_csv_row_error(str(exc)),
                    "message": f"Invalid CSV row {row_index}: {exc}",
                    "source_record_id": str(row_index),
                }
            )

    records, ingest_errors = canonical_records_from_payloads_isolated(
        row_payloads,
        source_type="csv",
        source_name=source_name,
    )
    if ingest_errors:
        for error in ingest_errors:
            if error.get("source_record_id") is None:
                continue
            error["message"] = f"Invalid CSV row {error['source_record_id']}: {error['message']}"
    return records, [*pre_ingest_errors, *ingest_errors]


def frames_from_csv(csv_text: str) -> list[TelemetryFrame]:
    records, _ = _records_from_csv_text_isolated(csv_text, source_name="csv_file")
    return frames_from_records(records)


def load_frames_from_json(
    path: str,
    *,
    return_ingest_errors: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p = Path(path)
    if not p.exists():
        raise SIIValidationError(f"Input file not found: {path}")
    if not p.is_file():
        raise SIIValidationError(f"Input path is not a file: {path}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SIIValidationError(f"Invalid JSON input: {path}") from exc
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise SIIValidationError("JSON input must be an object or an array of objects")
    records, ingest_errors = canonical_records_from_payloads_isolated(
        list(raw),
        source_type="json",
        source_name=str(p.name),
    )
    out = [ingestion_record_to_payload(record) for record in records]
    if return_ingest_errors:
        return out, ingest_errors
    return out


def load_frames_from_csv(
    path: str,
    *,
    return_ingest_errors: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p = Path(path)
    if not p.exists():
        raise SIIValidationError(f"Input file not found: {path}")
    if not p.is_file():
        raise SIIValidationError(f"Input path is not a file: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as exc:
        raise SIIValidationError(f"Failed to read CSV input: {path}") from exc
    records, ingest_errors = _records_from_csv_text_isolated(text, source_name=str(p.name))
    parsed = frames_from_records(records)
    out: list[dict[str, Any]] = []
    for f in parsed:
        sensor_values = dict(f.sensor_values)
        missingness = (
            float(sum(1 for v in sensor_values.values() if v is None)) / float(max(1, len(sensor_values)))
        )
        out.append(
            {
                "timestamp": f.timestamp,
                "site_id": f.site_id,
                "system_id": f.system_id,
                "asset_id": f.asset_id,
                "node_id": f.node_id,
                "variables": sensor_values,
                "sensor_values": dict(f.sensor_values),
                "quality_metadata": {
                    "missingness_rate": missingness,
                    "stale_signal": bool(f.quality_metadata.stale_signal),
                    "flatlined_variables": list(f.quality_metadata.flatlined_variables),
                    "source_status": str(f.quality_metadata.source_status),
                    "validation_issues": list(f.quality_metadata.validation_issues),
                    "completeness_score": float(f.quality_metadata.completeness_score),
                },
                "source_metadata": {
                    "source_type": f.source_metadata.source_type,
                    "source_name": f.source_metadata.source_name,
                    "source_record_id": f.source_metadata.source_record_id,
                    "source_schema_version": f.source_metadata.source_schema_version,
                    "request_id": f.source_metadata.request_id,
                    "tags": dict(f.source_metadata.tags),
                },
                "metadata": dict(f.metadata),
            }
        )
    if return_ingest_errors:
        return out, ingest_errors
    return out
