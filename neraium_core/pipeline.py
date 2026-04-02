<<<<<<< HEAD
from datetime import datetime, timezone
import numpy as np

from neraium_core.features import MicroFeatureEngine


class TelemetryPipeline:

    def __init__(self):

        self.features = MicroFeatureEngine()

        self.history = []
        self.baseline_mean = None
        self.baseline_cov = None

        self.training_samples = 50

    def _update_baseline(self, vector):

        self.history.append(vector)

        if len(self.history) < self.training_samples:
            return

        data = np.array(self.history[-self.training_samples:])

        self.baseline_mean = np.mean(data, axis=0)
        self.baseline_cov = np.cov(data, rowvar=False)

    def _mahalanobis(self, vector):

        if self.baseline_mean is None:
            return 0

        x = np.array(vector)

        diff = x - self.baseline_mean

        try:
            inv_cov = np.linalg.pinv(self.baseline_cov)
        except:
            return 0

        score = np.sqrt(diff.T @ inv_cov @ diff)

        return float(score)

    def process(self, payload):

        cpu = payload.signals["cpu_usage"]
        mem = payload.signals["memory_usage"]

        f = self.features.compute(cpu, mem)

        vector = [
            f["cpu"],
            f["memory"],
            f["cpu_delta"],
            f["mem_delta"],
            f["cpu_std"],
            f["mem_std"]
        ]

        self._update_baseline(vector)

        score = self._mahalanobis(vector)

        if score > 4:
            status = "anomaly"
        else:
            status = "normal"

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signals": {
                "cpu_usage": cpu,
                "memory_usage": mem
            },
            "score": score,

            "status": status
        }

        return event
=======
import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Dict, List, Mapping, Optional


DEFAULT_SITE_ID = "default-site"
DEFAULT_ASSET_ID = "default-asset"
DEFAULT_CUSTOMER_ID = "default-customer"
# Legacy: ingest no longer requires these literal header names — use semantic mapping in csv_mapping.py.
REQUIRED_CSV_COLUMNS = {"timestamp", "site_id", "asset_id"}


@dataclass(frozen=True)
class CsvIngestionIssue:
    code: str
    message: str
    severity: str = "error"
    row: int | None = None
    column: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalIngestionSignalRecord:
    """
    Canonical normalized ingest row for external formats before SII frame conversion.
    """

    timestamp: str
    asset_id: str
    site_id: str | None
    signals: dict[str, float | None]
    row_index: int


@dataclass(frozen=True)
class CsvSemanticMappingResult:
    mapping: Any | None
    issues: list[CsvIngestionIssue] = field(default_factory=list)
    warnings: list[CsvIngestionIssue] = field(default_factory=list)
    requires_confirmation: bool = False


@dataclass(frozen=True)
class CsvPipelineResult:
    canonical_records: list[CanonicalIngestionSignalRecord]
    issues: list[CsvIngestionIssue] = field(default_factory=list)
    warnings: list[CsvIngestionIssue] = field(default_factory=list)
    mapping: Any | None = None


def issue_to_dict(issue: CsvIngestionIssue) -> dict[str, Any]:
    return {
        "code": issue.code,
        "message": issue.message,
        "severity": issue.severity,
        "row": issue.row,
        "column": issue.column,
        "details": dict(issue.details),
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_timestamp(value: Any) -> str:
    """
    Normalize a timestamp into an ISO-8601 UTC string.
    Accepts datetime objects or strings. Falls back to current UTC time
    only when the input is None or empty.
    """
    if value is None or str(value).strip() == "":
        return now_iso()

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = None
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = None
        if dt is None:
            # Unix epoch seconds or milliseconds (common in exports)
            try:
                num = float(text)
                if 1e9 <= abs(num) <= 1e12:
                    dt = datetime.fromtimestamp(num, tz=timezone.utc)
                elif 1e12 < abs(num) <= 1e15:
                    dt = datetime.fromtimestamp(num / 1000.0, tz=timezone.utc)
            except (ValueError, OSError, OverflowError):
                dt = None
        if dt is None:
            raise ValueError(f"Invalid timestamp: {value!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc).isoformat()


def normalize_identifier(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_sensor_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Sensor name cannot be empty")
    return text


def pilot_hardening_enabled() -> bool:
    """
    Pilot hardening feature toggle.

    When enabled, the pipeline rejects non-numeric sensor values and treats NaN/inf
    as missing (`None`) to keep downstream analytics stable.
    """

    v = os.getenv("NERAIUM_PILOT_HARDENING", "0").strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def coerce_float(value: Any, *, sensor_name: str) -> Optional[float]:
    """
    Convert a sensor input value into a float.

    Returns:
      - `None` for missing values (`None`, empty string).
      - In pilot mode, rejects malformed non-numeric values with `ValueError`.
      - In pilot mode, converts NaN/inf to `None`.
    """

    strict = pilot_hardening_enabled()

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            f = float(text)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Invalid signal value for {sensor_name!r}: {value!r}") from exc
            return None
    elif isinstance(value, (int, float)):
        try:
            f = float(value)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(f"Invalid signal value for {sensor_name!r}: {value!r}") from exc
            return None
    else:
        if strict:
            raise ValueError(f"Invalid signal type for {sensor_name!r}: {type(value).__name__}")
        return None

    if strict and (math.isnan(f) or math.isinf(f)):
        return None

    return f


def build_frame(
    timestamp: Any,
    site_id: Any,
    asset_id: Any,
    sensor_values: Dict[Any, Any],
    customer_id: Any = DEFAULT_CUSTOMER_ID,
) -> Dict[str, Any]:
    """
    Build the internal telemetry frame for `StructuralEngine.process_frame()`.

    Internal contract:
    - `frame["timestamp"]` is an ISO-8601 UTC string
    - `frame["sensor_values"]` is a dict of `{signal_name: float | None}`
    """
    if not isinstance(sensor_values, dict):
        raise ValueError("sensor_values must be an object")

    # Internal frame shape used by `StructuralEngine.process_frame`.
    # Keep this stable across pipelines/entrypoints so production ingestion works.
    frame: Dict[str, Any] = {
        "timestamp": normalize_timestamp(timestamp),
        "customer_id": normalize_identifier(customer_id, DEFAULT_CUSTOMER_ID),
        "site_id": normalize_identifier(site_id, DEFAULT_SITE_ID),
        "asset_id": normalize_identifier(asset_id, DEFAULT_ASSET_ID),
        "sensor_values": {},
        "sensor_quality": {},
        "aligned": [],
        "anomaly": False,
    }

    for raw_key, raw_value in sensor_values.items():
        sensor_name = normalize_sensor_name(raw_key)
        numeric_value = coerce_float(raw_value, sensor_name=sensor_name)

        frame["sensor_values"][sensor_name] = numeric_value
        frame["sensor_quality"][sensor_name] = "ok" if numeric_value is not None else "missing"

    return frame


def normalize_rest_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize an incoming REST payload into the internal frame format.

    In pilot hardening mode (`NERAIUM_PILOT_HARDENING=1`), validation is strict:
    - `sensor_values` must be an object/dict
    - sensor values must be numeric or numeric strings (or `null`)
    - invalid values are rejected with clear `ValueError` messages
    """
    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")

    return build_frame(
        timestamp=payload.get("timestamp"),
        customer_id=payload.get("customer_id", DEFAULT_CUSTOMER_ID),
        site_id=payload.get("site_id", DEFAULT_SITE_ID),
        asset_id=payload.get("asset_id", DEFAULT_ASSET_ID),
        sensor_values=payload.get("sensor_values", {}),
    )


def parse_csv_text(
    csv_text: str,
    *,
    customer_id: str | None = None,
    column_mapping: dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Parse CSV text into a list of normalized internal frames.

    Column roles are resolved via semantic mapping (see :mod:`neraium_core.csv_mapping`):
    one timestamp column, one asset/entity column, optional site, and one or more sensor columns.

    If ``column_mapping`` is omitted, roles are inferred from headers (and sample values when needed).
    """
    if not isinstance(csv_text, str):
        raise ValueError("csv_text must be a string")

    resolved_customer = customer_id or DEFAULT_CUSTOMER_ID
    result = run_csv_ingestion_pipeline(
        csv_text,
        customer_id=resolved_customer,
        column_mapping=column_mapping,
    )
    blocking = [i for i in result.issues if i.severity == "error"]
    if blocking:
        top = blocking[0]
        if top.row is not None:
            raise ValueError(f"Invalid CSV row {top.row}: {top.message}")
        raise ValueError(top.message)
    return canonical_records_to_frames(result.canonical_records, customer_id=resolved_customer)


def parse_csv_rows(csv_text: str) -> tuple[list[str], list[dict[str, Any]], list[CsvIngestionIssue]]:
    reader = csv.DictReader(StringIO(csv_text), restkey="__extra_columns__", restval=None)
    if reader.fieldnames is None:
        return [], [], []
    headers = [str(h).strip() for h in reader.fieldnames if h is not None]
    rows: list[dict[str, Any]] = []
    issues: list[CsvIngestionIssue] = []
    for row_index, row in enumerate(reader, start=2):
        if row is None:
            issues.append(
                CsvIngestionIssue(
                    code="malformed_row",
                    message="Malformed CSV row could not be parsed.",
                    row=row_index,
                )
            )
            continue
        extras = row.get("__extra_columns__")
        if extras:
            issues.append(
                CsvIngestionIssue(
                    code="malformed_row",
                    message="Row has more values than header columns.",
                    row=row_index,
                    details={"extra_values": [str(x) for x in extras]},
                )
            )
        cleaned = {str(k): v for k, v in row.items() if k != "__extra_columns__" and k is not None}
        cleaned["__row_index__"] = row_index
        rows.append(cleaned)
    return headers, rows, issues


def infer_csv_mapping_stage(
    headers: list[str],
    *,
    rows: list[Mapping[str, Any]],
    column_mapping: dict[str, Any] | None = None,
) -> CsvSemanticMappingResult:
    from neraium_core.csv_mapping import infer_semantic_mapping, resolve_mapping

    if not headers:
        return CsvSemanticMappingResult(
            mapping=None,
            issues=[CsvIngestionIssue(code="missing_header", message="CSV has no header row.")],
            requires_confirmation=True,
        )
    if column_mapping is not None:
        try:
            mapping, warnings = resolve_mapping(headers, column_mapping, csv_sample=None)
        except ValueError as exc:
            return CsvSemanticMappingResult(
                mapping=None,
                issues=[CsvIngestionIssue(code="invalid_mapping_override", message=str(exc))],
                requires_confirmation=True,
            )
        return CsvSemanticMappingResult(
            mapping=mapping,
            warnings=[CsvIngestionIssue(code="mapping_override", message=w, severity="warning") for w in warnings],
            requires_confirmation=False,
        )

    sample_rows = [{k: "" if v is None else str(v) for k, v in row.items() if not k.startswith("__")} for row in rows[:16]]
    inferred, issues, _debug = infer_semantic_mapping(headers, sample_rows=sample_rows)
    warning_issues = [CsvIngestionIssue(code="ambiguous_mapping", message=i, severity="warning") for i in issues if "Confirm" in i or "Multiple" in i]
    blocking = [CsvIngestionIssue(code="mapping_validation_error", message=i) for i in issues if i not in {w.message for w in warning_issues}]
    return CsvSemanticMappingResult(
        mapping=inferred,
        issues=blocking,
        warnings=warning_issues,
        requires_confirmation=inferred is None or bool(warning_issues),
    )


def validate_csv_mapping_stage(mapping: Any, headers: list[str]) -> list[CsvIngestionIssue]:
    from neraium_core.csv_mapping import validate_mapping

    errs = validate_mapping(mapping, headers)
    out: list[CsvIngestionIssue] = []
    for err in errs:
        code = "mapping_error"
        if "timestamp" in err.lower():
            code = "missing_timestamp"
        elif "asset" in err.lower():
            code = "missing_asset_id"
        elif "sensor" in err.lower():
            code = "no_usable_signal_columns"
        out.append(CsvIngestionIssue(code=code, message=err))
    return out


def normalize_csv_rows_to_canonical(
    rows: list[Mapping[str, Any]],
    *,
    mapping: Any,
) -> tuple[list[CanonicalIngestionSignalRecord], list[CsvIngestionIssue]]:
    canonical_records: list[CanonicalIngestionSignalRecord] = []
    issues: list[CsvIngestionIssue] = []
    for row in rows:
        row_index = int(row.get("__row_index__", 0) or 0)
        raw_timestamp = row.get(mapping.timestamp)
        try:
            ts = normalize_timestamp(raw_timestamp)
        except ValueError:
            issues.append(
                CsvIngestionIssue(
                    code="invalid_timestamp",
                    message=f"Invalid timestamp: {raw_timestamp!r}",
                    row=row_index or None,
                    column=mapping.timestamp,
                )
            )
            continue
        raw_asset = row.get(mapping.asset_id)
        asset_id = normalize_identifier(raw_asset, DEFAULT_ASSET_ID)
        if asset_id == DEFAULT_ASSET_ID:
            issues.append(
                CsvIngestionIssue(
                    code="missing_asset_id",
                    message="Missing asset/entity identifier.",
                    row=row_index or None,
                    column=mapping.asset_id,
                )
            )
            continue
        site_id = normalize_identifier(row.get(mapping.site_id), DEFAULT_SITE_ID) if mapping.site_id else None
        signals: dict[str, float | None] = {}
        usable_signal_count = 0
        for column in mapping.sensor_columns:
            raw_signal = row.get(column)
            try:
                numeric = coerce_float(raw_signal, sensor_name=column)
            except ValueError:
                issues.append(
                    CsvIngestionIssue(
                        code="non_numeric_signal",
                        message=f"Non-numeric signal value for {column!r}.",
                        row=row_index or None,
                        column=column,
                    )
                )
                numeric = None
            if numeric is None and raw_signal not in (None, ""):
                text = str(raw_signal).strip()
                if text != "":
                    try:
                        parsed = float(text)
                        if math.isnan(parsed) or math.isinf(parsed):
                            issues.append(
                                CsvIngestionIssue(
                                    code="non_numeric_signal",
                                    message=f"Non-finite signal value for {column!r}.",
                                    row=row_index or None,
                                    column=column,
                                )
                            )
                    except (TypeError, ValueError):
                        issues.append(
                            CsvIngestionIssue(
                                code="non_numeric_signal",
                                message=f"Non-numeric signal value for {column!r}.",
                                row=row_index or None,
                                column=column,
                            )
                        )
            if numeric is not None:
                usable_signal_count += 1
            signals[column] = numeric
        if not signals or usable_signal_count == 0:
            issues.append(
                CsvIngestionIssue(
                    code="no_usable_signal_values",
                    message="No usable numeric signal values in row.",
                    row=row_index or None,
                )
            )
            continue
        canonical_records.append(
            CanonicalIngestionSignalRecord(
                timestamp=ts,
                asset_id=asset_id,
                site_id=site_id,
                signals=signals,
                row_index=row_index or -1,
            )
        )
    return canonical_records, issues


def canonical_records_to_frames(
    records: list[CanonicalIngestionSignalRecord],
    *,
    customer_id: str,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for record in records:
        frames.append(
            build_frame(
                timestamp=record.timestamp,
                customer_id=customer_id,
                site_id=record.site_id if record.site_id is not None else DEFAULT_SITE_ID,
                asset_id=record.asset_id,
                sensor_values=record.signals,
            )
        )
    return frames


def run_csv_ingestion_pipeline(
    csv_text: str,
    *,
    customer_id: str,
    column_mapping: dict[str, Any] | None = None,
) -> CsvPipelineResult:
    headers, rows, parse_issues = parse_csv_rows(csv_text)
    map_result = infer_csv_mapping_stage(headers, rows=rows, column_mapping=column_mapping)
    issues = list(parse_issues) + list(map_result.issues)
    warnings = list(map_result.warnings)
    if map_result.mapping is None:
        return CsvPipelineResult(canonical_records=[], issues=issues, warnings=warnings, mapping=None)
    mapping_issues = validate_csv_mapping_stage(map_result.mapping, headers)
    issues.extend(mapping_issues)
    if any(i.severity == "error" for i in mapping_issues):
        return CsvPipelineResult(canonical_records=[], issues=issues, warnings=warnings, mapping=map_result.mapping)
    canonical_records, normalize_issues = normalize_csv_rows_to_canonical(rows, mapping=map_result.mapping)
    issues.extend(normalize_issues)
    if not canonical_records and rows:
        issues.append(
            CsvIngestionIssue(
                code="all_rows_invalid",
                message="All CSV rows failed normalization. Fix mapping or row values and retry.",
            )
        )
    return CsvPipelineResult(
        canonical_records=canonical_records,
        issues=issues,
        warnings=warnings,
        mapping=map_result.mapping,
    )
>>>>>>> b5a6787d331053eaea92f461f7bbab489f4c495a
