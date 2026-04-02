from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .pipeline import (
    DEFAULT_ASSET_ID,
    DEFAULT_SITE_ID,
    CanonicalIngestionSignalRecord,
    build_frame,
    canonical_records_to_frames,
    coerce_float,
    normalize_identifier,
    normalize_timestamp,
)


@dataclass(frozen=True)
class MappingReview:
    inferred_mapping: dict[str, Any]
    warnings: list[str]
    requires_confirmation: bool


_RESERVED_TOP_LEVEL = {
    "timestamp",
    "time",
    "ts",
    "recorded_at",
    "event_time",
    "asset_id",
    "asset",
    "entity_id",
    "entity",
    "device_id",
    "machine",
    "unit",
    "site_id",
    "site",
    "system_id",
    "location",
    "plant",
    "customer_id",
    "signals",
    "sensor_values",
    "variables",
    "measurements",
    "values",
    "metadata",
}

_TIMESTAMP_KEYS = ("timestamp", "time", "ts", "recorded_at", "event_time")
_ASSET_KEYS = ("asset_id", "asset", "entity_id", "entity", "device_id", "machine", "unit")
_SITE_KEYS = ("site_id", "site", "system_id", "location", "plant")
_SIGNAL_KEYS = ("signals", "sensor_values", "variables", "measurements", "values")


def _pick_first(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None and str(payload.get(key)).strip() != "":
            return payload.get(key)
    return None


def _pick_first_key(payload: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in payload and payload.get(key) is not None and str(payload.get(key)).strip() != "":
            return key
    return None


def _present_keys(payload: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    return [k for k in keys if k in payload and payload.get(k) is not None and str(payload.get(k)).strip() != ""]


def _pick_mapped_or_inferred(
    payload: Mapping[str, Any],
    *,
    mapped_key: str,
    aliases: Sequence[str],
    role: str,
) -> Any:
    explicit = str(mapped_key or "").strip()
    if explicit:
        if explicit not in payload:
            raise ValueError(f"missing_{role}: mapped field {explicit!r} was not found in payload")
        return payload.get(explicit)

    candidates = _present_keys(payload, aliases)
    if len(candidates) > 1:
        raise ValueError(
            f"ambiguous_mapping: multiple candidate {role} fields found: {', '.join(candidates)}. "
            f"Provide explicit mapping for {role}."
        )
    if len(candidates) == 1:
        return payload.get(candidates[0])
    return None


def _normalize_signal_name(name: Any, *, signal_aliases: Mapping[str, str] | None = None) -> str:
    raw = str(name).strip()
    if not raw:
        raise ValueError("validation_error: empty signal field name")
    if not signal_aliases:
        return raw
    mapped = signal_aliases.get(raw)
    if mapped is None:
        return raw
    text = str(mapped).strip()
    if not text:
        raise ValueError(f"validation_error: mapped signal name is empty for field {raw!r}")
    return text


def _extract_signal_map(
    payload: Mapping[str, Any],
    *,
    signal_aliases: Mapping[str, str] | None = None,
    mapping: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    mapping = dict(mapping or {})
    explicit_signals_key = str(mapping.get("signals") or mapping.get("signals_key") or "").strip()
    if explicit_signals_key:
        nested = payload.get(explicit_signals_key)
    else:
        nested = _pick_first(payload, _SIGNAL_KEYS)

    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("unsupported_payload_shape: signals/sensor_values must be an object")
        out: dict[str, Any] = {}
        for k, v in nested.items():
            out[_normalize_signal_name(k, signal_aliases=signal_aliases)] = v
        return out

    explicit_sensor_fields = mapping.get("sensor_fields")
    sensor_field_names: list[str] = []
    if isinstance(explicit_sensor_fields, list):
        sensor_field_names = [str(x).strip() for x in explicit_sensor_fields if str(x).strip()]

    out: dict[str, Any] = {}
    if sensor_field_names:
        for key in sensor_field_names:
            if key in payload:
                out[_normalize_signal_name(key, signal_aliases=signal_aliases)] = payload.get(key)
        return out

    for key, value in payload.items():
        if str(key).strip().lower() in _RESERVED_TOP_LEVEL:
            continue
        out[_normalize_signal_name(key, signal_aliases=signal_aliases)] = value
    return out


def infer_payload_mapping(payload: Mapping[str, Any]) -> MappingReview:
    timestamp_key = _pick_first_key(payload, _TIMESTAMP_KEYS)
    asset_key = _pick_first_key(payload, _ASSET_KEYS)
    site_key = _pick_first_key(payload, _SITE_KEYS)
    signals_key = _pick_first_key(payload, _SIGNAL_KEYS)
    inferred = {
        "timestamp": timestamp_key,
        "asset_id": asset_key,
        "site_id": site_key,
        "signals": signals_key,
    }
    warnings: list[str] = []
    if timestamp_key is None:
        warnings.append("missing_timestamp: could not infer timestamp alias")
    if asset_key is None:
        warnings.append("missing_asset_id: could not infer asset/entity identifier alias")
    if len(_present_keys(payload, _TIMESTAMP_KEYS)) > 1:
        warnings.append("ambiguous_mapping: multiple timestamp aliases found")
    if len(_present_keys(payload, _ASSET_KEYS)) > 1:
        warnings.append("ambiguous_mapping: multiple asset aliases found")

    requires_confirmation = False
    if signals_key is None:
        candidate_keys = [k for k in payload.keys() if str(k).strip().lower() not in _RESERVED_TOP_LEVEL]
        if not candidate_keys:
            warnings.append("no_usable_signals: payload has no signals object and no signal-like fields")
            requires_confirmation = True
    return MappingReview(
        inferred_mapping=inferred,
        warnings=warnings,
        requires_confirmation=requires_confirmation or bool(warnings),
    )


def _canonical_from_payload(
    payload: Mapping[str, Any],
    *,
    mapping: Mapping[str, Any] | None,
    customer_id: str | None,
    row_index: int,
) -> CanonicalIngestionSignalRecord:
    mapping = dict(mapping or {})
    signal_aliases_raw = mapping.get("signal_aliases")
    signal_aliases = (
        {str(k): str(v) for k, v in signal_aliases_raw.items()} if isinstance(signal_aliases_raw, Mapping) else None
    )

    timestamp_val = _pick_mapped_or_inferred(
        payload,
        mapped_key=str(mapping.get("timestamp") or ""),
        aliases=_TIMESTAMP_KEYS,
        role="timestamp",
    )
    asset_val = _pick_mapped_or_inferred(
        payload,
        mapped_key=str(mapping.get("asset_id") or ""),
        aliases=_ASSET_KEYS,
        role="asset_id",
    )
    site_val = _pick_mapped_or_inferred(
        payload,
        mapped_key=str(mapping.get("site_id") or ""),
        aliases=_SITE_KEYS,
        role="site_id",
    )

    if timestamp_val is None:
        raise ValueError("missing_timestamp: provide timestamp/time/recorded_at or explicit mapping.timestamp")
    if asset_val is None:
        raise ValueError("missing_asset_id: provide asset_id/entity_id or explicit mapping.asset_id")

    signal_values = _extract_signal_map(payload, signal_aliases=signal_aliases, mapping=mapping)
    if not signal_values:
        raise ValueError("no_usable_signals: provide signals/sensor_values object or signal fields")

    normalized_signals: dict[str, float | None] = {}
    usable = 0
    for key, value in signal_values.items():
        numeric = coerce_float(value, sensor_name=key)
        if numeric is not None:
            usable += 1
        normalized_signals[key] = numeric
    if usable == 0:
        raise ValueError("invalid_signal_values: no usable numeric signals found")

    return CanonicalIngestionSignalRecord(
        timestamp=normalize_timestamp(timestamp_val),
        asset_id=normalize_identifier(asset_val, DEFAULT_ASSET_ID),
        site_id=normalize_identifier(site_val, DEFAULT_SITE_ID) if site_val is not None else None,
        signals=normalized_signals,
        row_index=row_index,
    )


def normalize_external_payload(
    payload: Mapping[str, Any],
    *,
    customer_id: str | None = None,
    mapping: Mapping[str, Any] | None = None,
    require_confirmed_mapping: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("unsupported_payload_shape: payload must be an object")

    review = infer_payload_mapping(payload)
    if require_confirmed_mapping and review.requires_confirmation:
        raise ValueError(
            "ambiguous_mapping: external field mapping needs confirmation before ingest. "
            + "; ".join(review.warnings)
        )

    canonical = _canonical_from_payload(payload, mapping=mapping, customer_id=customer_id, row_index=1)
    frame = canonical_records_to_frames([canonical], customer_id=customer_id or "default-customer")[0]
    return frame


def _extract_json_items(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise ValueError("unsupported_payload_shape: expected object or list")

    for key in ("items", "records", "payloads"):
        candidate = payload.get(key)
        if candidate is not None:
            if not isinstance(candidate, list):
                raise ValueError("unsupported_payload_shape: batch records must be a list")
            return candidate

    stream = payload.get("stream")
    if isinstance(stream, Mapping):
        candidate = stream.get("records")
        if candidate is not None:
            if not isinstance(candidate, list):
                raise ValueError("unsupported_payload_shape: stream.records must be a list")
            return candidate

    return [payload]


def normalize_external_batch_payload(
    payload: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    customer_id: str | None = None,
) -> tuple[list[dict[str, Any]], MappingReview]:
    items = _extract_json_items(payload)
    mapping = payload.get("mapping") if isinstance(payload, Mapping) and isinstance(payload.get("mapping"), Mapping) else {}

    review = MappingReview(
        inferred_mapping={"mapping": dict(mapping)},
        warnings=[],
        requires_confirmation=False,
    )
    canonical_records: list[CanonicalIngestionSignalRecord] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"batch_item_{idx}: unsupported_payload_shape: item must be an object")
        try:
            canonical_records.append(
                _canonical_from_payload(
                    item,
                    mapping=mapping,
                    customer_id=customer_id,
                    row_index=idx,
                )
            )
        except ValueError as exc:
            raise ValueError(f"batch_item_{idx}: {exc}") from exc

    frames = canonical_records_to_frames(canonical_records, customer_id=customer_id or "default-customer")
    return frames, review


def normalize_canonical_records_payload(
    payload: Mapping[str, Any] | list[Mapping[str, Any]],
    *,
    customer_id: str | None = None,
) -> list[dict[str, Any]]:
    items = _extract_json_items(payload)
    canonical_records: list[CanonicalIngestionSignalRecord] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"batch_item_{idx}: unsupported_payload_shape: canonical item must be object")
        if "signals" not in item or "timestamp" not in item or "asset_id" not in item:
            raise ValueError(f"batch_item_{idx}: canonical_schema_violation: timestamp, asset_id, and signals are required")
        signals = item.get("signals")
        if not isinstance(signals, Mapping):
            raise ValueError(f"batch_item_{idx}: canonical_schema_violation: signals must be an object")
        cleaned_signals: dict[str, float | None] = {}
        usable = 0
        for name, value in signals.items():
            signal_name = str(name).strip()
            if not signal_name:
                raise ValueError(f"batch_item_{idx}: canonical_schema_violation: signal names must be non-empty")
            numeric = coerce_float(value, sensor_name=signal_name)
            if numeric is not None:
                usable += 1
            cleaned_signals[signal_name] = numeric
        if usable == 0:
            raise ValueError(f"batch_item_{idx}: invalid_signal_values: no usable numeric signals found")

        canonical_records.append(
            CanonicalIngestionSignalRecord(
                timestamp=normalize_timestamp(item.get("timestamp")),
                asset_id=normalize_identifier(item.get("asset_id"), DEFAULT_ASSET_ID),
                site_id=normalize_identifier(item.get("site_id"), DEFAULT_SITE_ID)
                if item.get("site_id") is not None
                else None,
                signals=cleaned_signals,
                row_index=int(item.get("row_index") or idx),
            )
        )
    return canonical_records_to_frames(canonical_records, customer_id=customer_id or "default-customer")
