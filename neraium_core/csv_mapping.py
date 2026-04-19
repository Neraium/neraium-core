"""
Semantic CSV column mapping: infer roles from arbitrary headers, validate, and map rows
to the canonical ingest frame shape (timestamp, site_id, asset_id, sensor_values).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# --- Header heuristics (normalized: lower, collapsed spaces) ---

_TIMESTAMP_EXACT = frozenset(
    {
        "timestamp",
        "time",
        "ts",
        "utc",
        "datetime",
        "date_time",
        "recorded_at",
        "recorded",
        "when",
    }
)
_TIMESTAMP_SUBSTR = ("time", "date", "ts", "utc", "stamp", "recorded", "observed")

_SITE_EXACT = frozenset({"site_id", "site", "plant", "location", "facility", "yard", "area", "zone"})
_SITE_SUBSTR = ("site", "plant", "location", "facility", "yard", "line", "area", "zone", "region")

_ASSET_EXACT = frozenset(
    {
        "asset_id",
        "asset",
        "unit",
        "equipment",
        "machine",
        "device",
        "tag",
        "entity",
        "compressor",
        "pump",
        "motor",
    }
)
_ASSET_SUBSTR = ("asset", "unit", "equipment", "machine", "device", "tag", "compressor", "pump", "turbine")

_SENSOR_HINT_SUBSTR = (
    "pressure",
    "flow",
    "temp",
    "vibration",
    "vibe",
    "rpm",
    "speed",
    "load",
    "current",
    "power",
    "torque",
    "humidity",
    "accel",
    "displacement",
    "sensor",
    "signal",
    "measurement",
    "reading",
    "value",
    "channel",
)


def _norm_header(h: str) -> str:
    return " ".join(str(h).strip().lower().split())


def _score_timestamp_header(h: str) -> float:
    n = _norm_header(h)
    if n in _TIMESTAMP_EXACT:
        return 1.0
    if any(n == p or n.endswith("_" + p) for p in ("time", "at", "ts")):
        return 0.82
    s = 0.0
    for sub in _TIMESTAMP_SUBSTR:
        if sub in n:
            s = max(s, 0.55 + 0.1 * (len(sub) / 12.0))
    if re.search(r"\b(t|dt)\d*\b", n):
        s = max(s, 0.45)
    return min(1.0, s)


def _score_site_header(h: str) -> float:
    n = _norm_header(h)
    if n in _SITE_EXACT:
        return 1.0
    s = 0.0
    for sub in _SITE_SUBSTR:
        if sub in n:
            s = max(s, 0.72)
    return min(1.0, s)


def _score_asset_header(h: str) -> float:
    n = _norm_header(h)
    if n in _ASSET_EXACT:
        return 1.0
    # "id" alone is ambiguous — weak signal
    if n in ("id", "row", "row_id", "index"):
        return 0.15
    s = 0.0
    for sub in _ASSET_SUBSTR:
        if sub in n:
            s = max(s, 0.78)
    if re.search(r"\b(tag|name|label)\b", n) and "site" not in n and "time" not in n:
        s = max(s, 0.5)
    return min(1.0, s)


def _score_sensor_header(h: str) -> float:
    n = _norm_header(h)
    if any(k in n for k in _SENSOR_HINT_SUBSTR):
        return 0.85
    # generic short names often numeric channels
    if re.match(r"^[sv]\d+$", n) or re.match(r"^x\d+$", n) or re.match(r"^ch\d+$", n):
        return 0.7
    return 0.25


def parse_csv_sample_for_mapping(csv_text: str, max_rows: int = 16) -> Tuple[List[str], List[Dict[str, str]]]:
    """Return header list and first data rows from a CSV snippet (for preview / inference)."""
    return _parse_sample_rows(csv_text, max_rows=max_rows)


def _parse_sample_rows(csv_text: str, max_rows: int = 8) -> Tuple[List[str], List[Dict[str, str]]]:
    reader = csv.DictReader(StringIO(csv_text))
    if reader.fieldnames is None:
        return [], []
    headers = [str(h).strip() for h in reader.fieldnames if h is not None]
    rows: List[Dict[str, str]] = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        if row:
            rows.append({str(k): ("" if v is None else str(v)) for k, v in row.items() if k is not None})
    return headers, rows


def _boost_timestamp_from_samples(headers: List[str], sample_rows: List[Mapping[str, str]]) -> Dict[str, float]:
    """Increase score for columns whose sample values look like ISO timestamps or epoch numbers."""
    boost: Dict[str, float] = {h: 0.0 for h in headers}
    iso_re = re.compile(
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$",
        re.I,
    )
    for h in headers:
        hits = 0
        total = 0
        for row in sample_rows:
            if h not in row:
                continue
            total += 1
            v = str(row.get(h, "")).strip()
            if not v:
                continue
            if iso_re.match(v):
                hits += 1
                continue
            try:
                fv = float(v)
                if 1e9 <= abs(fv) <= 1e12 or 1e12 <= abs(fv) <= 1e15:
                    hits += 1
            except ValueError:
                pass
        if total > 0 and hits / total >= 0.5:
            boost[h] = 0.35 + 0.2 * (hits / total)
    return boost


@dataclass
class SemanticColumnMapping:
    """Maps CSV header names to canonical roles."""

    timestamp: str
    asset_id: str
    site_id: Optional[str] = None
    sensor_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "asset_id": self.asset_id,
            "site_id": self.site_id,
            "sensor_columns": list(self.sensor_columns),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticColumnMapping:
        ts = str(data.get("timestamp") or "").strip()
        asset = str(data.get("asset_id") or "").strip()
        site_raw = data.get("site_id")
        site = str(site_raw).strip() if site_raw not in (None, "") else None
        sensors = data.get("sensor_columns") or []
        if not isinstance(sensors, list):
            raise ValueError("sensor_columns must be a list of column names")
        return cls(
            timestamp=ts,
            asset_id=asset,
            site_id=site,
            sensor_columns=[str(s).strip() for s in sensors if str(s).strip()],
        )


def infer_semantic_mapping(
    headers: Sequence[str],
    *,
    sample_rows: Optional[Sequence[Mapping[str, str]]] = None,
) -> Tuple[Optional[SemanticColumnMapping], List[str], Dict[str, Any]]:
    """
    Infer timestamp, asset, optional site, and sensor columns from arbitrary headers.

    Returns (mapping or None, issues, debug info).
    """
    raw = [str(h).strip() for h in headers if str(h).strip()]
    if not raw:
        return None, ["CSV has no header columns."], {}

    # Preserve first occurrence order; detect duplicates
    seen: set[str] = set()
    unique: List[str] = []
    for h in raw:
        if h in seen:
            continue
        seen.add(h)
        unique.append(h)

    ts_boost: Dict[str, float] = {}
    if sample_rows:
        ts_boost = _boost_timestamp_from_samples(unique, list(sample_rows))

    ts_scores = {h: _score_timestamp_header(h) + ts_boost.get(h, 0.0) for h in unique}
    site_scores = {h: _score_site_header(h) for h in unique}
    asset_scores = {h: _score_asset_header(h) for h in unique}
    sensor_scores = {h: _score_sensor_header(h) for h in unique}

    issues: List[str] = []

    # Pick timestamp
    best_ts = max(unique, key=lambda h: ts_scores.get(h, 0.0)) if unique else ""
    if not best_ts or ts_scores.get(best_ts, 0.0) < 0.28:
        issues.append(
            "Could not confidently identify a time column. Pick the column that contains "
            "timestamps (ISO-8601 or epoch seconds/milliseconds)."
        )
        best_ts = ""

    pool = [h for h in unique if h != best_ts]

    # Asset: strongest among pool
    best_asset = ""
    if pool:
        best_asset = max(pool, key=lambda h: asset_scores.get(h, 0.0))
        if asset_scores.get(best_asset, 0.0) < 0.22:
            issues.append(
                "Could not confidently identify an asset / entity column. "
                "Pick one column that identifies the equipment or entity for each row."
            )
            best_asset = ""
        else:
            pool = [h for h in pool if h != best_asset]

    # Site (optional)
    best_site: Optional[str] = None
    if pool:
        site_candidate = max(pool, key=lambda h: site_scores.get(h, 0.0))
        if site_scores.get(site_candidate, 0.0) >= 0.35:
            best_site = site_candidate
            pool = [h for h in pool if h != site_candidate]

    # Remaining columns default to sensors
    sensors = list(pool)
    if not sensors:
        issues.append("Need at least one sensor / measurement column (numeric telemetry).")

    # If we still have no sensors but had unassigned high sensor-score columns tied to ts/asset — rare
    debug = {
        "scores": {
            "timestamp": ts_scores,
            "site": site_scores,
            "asset": asset_scores,
            "sensor_hint": sensor_scores,
        },
        "picked": {"timestamp": best_ts, "asset_id": best_asset, "site_id": best_site},
    }

    if not best_ts or not best_asset or not sensors:
        return None, issues, debug

    # Ambiguity warnings (non-fatal)
    ts_ties = [h for h in unique if h != best_ts and ts_scores.get(h, 0.0) >= ts_scores.get(best_ts, 0.0) - 0.08]
    if len(ts_ties) > 1:
        issues.append(
            "Multiple columns resemble timestamps. Confirm the correct time column in the mapping UI."
        )

    mapping = SemanticColumnMapping(
        timestamp=best_ts,
        asset_id=best_asset,
        site_id=best_site,
        sensor_columns=sensors,
    )
    return mapping, issues, debug


def validate_mapping(mapping: SemanticColumnMapping, headers: Sequence[str]) -> List[str]:
    errs: List[str] = []
    header_set = {str(h).strip() for h in headers}
    if not mapping.timestamp:
        errs.append("Mapping requires a timestamp column.")
    elif mapping.timestamp not in header_set:
        errs.append(f"Timestamp column {mapping.timestamp!r} is not present in the CSV header.")

    if not mapping.asset_id:
        errs.append("Mapping requires an asset / entity identifier column.")
    elif mapping.asset_id not in header_set:
        errs.append(f"Asset column {mapping.asset_id!r} is not present in the CSV header.")

    if mapping.site_id and mapping.site_id not in header_set:
        errs.append(f"Site column {mapping.site_id!r} is not present in the CSV header.")

    if not mapping.sensor_columns or len(mapping.sensor_columns) < 1:
        errs.append("Select at least one sensor / numeric telemetry column.")
        return errs

    reserved = {mapping.timestamp, mapping.asset_id}
    if mapping.site_id:
        reserved.add(mapping.site_id)
    for col in mapping.sensor_columns:
        if col not in header_set:
            errs.append(f"Sensor column {col!r} is not present in the CSV header.")
        if col in reserved:
            errs.append(f"Column {col!r} cannot be both a key field and a sensor.")

    # dedupe sensors
    if len(set(mapping.sensor_columns)) != len(mapping.sensor_columns):
        errs.append("Duplicate sensor column entries in mapping.")

    return errs


def row_to_frame_kwargs(
    row: Mapping[str, Any],
    mapping: SemanticColumnMapping,
    *,
    customer_id: str,
) -> Dict[str, Any]:
    """Build kwargs for pipeline.build_frame from one CSV row dict."""
    sensor_values: Dict[str, Any] = {}
    for col in mapping.sensor_columns:
        sensor_values[col] = row.get(col)
    site_val = row.get(mapping.site_id) if mapping.site_id else None
    return {
        "timestamp": row.get(mapping.timestamp),
        "customer_id": customer_id,
        "site_id": site_val,
        "asset_id": row.get(mapping.asset_id),
        "sensor_values": sensor_values,
    }


def resolve_mapping(
    headers: Sequence[str],
    column_mapping: Mapping[str, Any] | SemanticColumnMapping | None,
    *,
    csv_sample: str | None = None,
    require_confirmation_for_ambiguous: bool = True,
) -> Tuple[SemanticColumnMapping, List[str]]:
    """
    Resolve final mapping: explicit user mapping wins; otherwise infer from headers (+ optional sample rows).
    Returns (mapping, warnings).
    """
    headers_list = [str(h).strip() for h in headers if h is not None and str(h).strip()]
    sample_rows: Optional[List[Mapping[str, str]]] = None
    if csv_sample:
        _, sample_rows = _parse_sample_rows(csv_sample, max_rows=12)

    if column_mapping is not None:
        if isinstance(column_mapping, SemanticColumnMapping):
            m = column_mapping
        else:
            m = SemanticColumnMapping.from_dict(column_mapping)
        errs = validate_mapping(m, headers_list)
        if errs:
            raise ValueError("; ".join(errs))
        return m, []

    inferred, issues, _ = infer_semantic_mapping(headers_list, sample_rows=sample_rows)
    if inferred is None:
        raise ValueError(
            "Could not infer CSV column roles automatically. "
            + (" ".join(issues) if issues else "Provide a column_mapping payload.")
        )
    # issues may contain ambiguity warnings — surface as non-fatal
    warnings = [i for i in issues if "Confirm" in i or "Multiple" in i]
    if warnings and require_confirmation_for_ambiguous:
        raise ValueError(
            "ambiguous_mapping: inferred CSV mapping needs operator confirmation. "
            + " ".join(warnings)
        )
    return inferred, warnings
