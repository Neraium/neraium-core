from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .pipeline import DEFAULT_ASSET_ID, DEFAULT_SITE_ID, build_frame


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


def _pick_first(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload.get(key) is not None and str(payload.get(key)).strip() != "":
            return payload.get(key)
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
) -> dict[str, Any]:
    nested = _pick_first(payload, ("signals", "sensor_values", "variables", "measurements", "values"))
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("unsupported_payload_shape: signals/sensor_values must be an object")
        out: dict[str, Any] = {}
        for k, v in nested.items():
            out[_normalize_signal_name(k, signal_aliases=signal_aliases)] = v
        return out

    out: dict[str, Any] = {}
    for key, value in payload.items():
        if str(key).strip().lower() in _RESERVED_TOP_LEVEL:
            continue
        out[_normalize_signal_name(key, signal_aliases=signal_aliases)] = value
    return out


def infer_payload_mapping(payload: Mapping[str, Any]) -> MappingReview:
    inferred = {
        "timestamp": _pick_first(payload, ("timestamp", "time", "ts", "recorded_at", "event_time")),
        "asset_id": _pick_first(payload, ("asset_id", "asset", "entity_id", "entity", "device_id", "machine", "unit")),
        "site_id": _pick_first(payload, ("site_id", "site", "system_id", "location", "plant")),
        "signals_key": _pick_first(payload, ("signals", "sensor_values", "variables", "measurements", "values")),
    }
    warnings: list[str] = []
    if inferred["timestamp"] is None:
        warnings.append("missing_timestamp: could not infer timestamp alias")
    if inferred["asset_id"] is None:
        warnings.append("missing_asset_id: could not infer asset/entity identifier alias")
    requires_confirmation = False
    if inferred["signals_key"] is None:
        candidate_keys = [k for k in payload.keys() if str(k).strip().lower() not in _RESERVED_TOP_LEVEL]
        if not candidate_keys:
            warnings.append("no_usable_signals: payload has no signals object and no signal-like fields")
            requires_confirmation = True
    return MappingReview(
        inferred_mapping={
            "timestamp": "timestamp|time|ts|recorded_at|event_time",
            "asset_id": "asset_id|asset|entity_id|entity|device_id|machine|unit",
            "site_id": "site_id|site|system_id|location|plant",
            "signals": "signals|sensor_values|variables|measurements|values|top-level numeric-like fields",
        },
        warnings=warnings,
        requires_confirmation=requires_confirmation or bool(warnings),
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

    mapping = dict(mapping or {})
    review = infer_payload_mapping(payload)
    if require_confirmed_mapping and review.requires_confirmation:
        raise ValueError(
            "ambiguous_mapping: external field mapping needs confirmation before ingest. "
            + "; ".join(review.warnings)
        )

    signal_aliases_raw = mapping.get("signal_aliases")
    signal_aliases = (
        {str(k): str(v) for k, v in signal_aliases_raw.items()} if isinstance(signal_aliases_raw, Mapping) else None
    )

    timestamp_key = str(mapping.get("timestamp") or "").strip()
    asset_key = str(mapping.get("asset_id") or "").strip()
    site_key = str(mapping.get("site_id") or "").strip()

    timestamp_val = payload.get(timestamp_key) if timestamp_key else _pick_first(payload, ("timestamp", "time", "ts", "recorded_at", "event_time"))
    asset_val = payload.get(asset_key) if asset_key else _pick_first(payload, ("asset_id", "asset", "entity_id", "entity", "device_id", "machine", "unit"))
    site_val = payload.get(site_key) if site_key else _pick_first(payload, ("site_id", "site", "system_id", "location", "plant"))

    if timestamp_val is None:
        raise ValueError("missing_timestamp: provide timestamp/time/recorded_at or explicit mapping.timestamp")
    if asset_val is None:
        raise ValueError("missing_asset_id: provide asset_id/entity_id or explicit mapping.asset_id")

    signal_values = _extract_signal_map(payload, signal_aliases=signal_aliases)
    if not signal_values:
        raise ValueError("no_usable_signals: provide signals/sensor_values object or signal fields")

    frame = build_frame(
        timestamp=timestamp_val,
        site_id=site_val if site_val is not None else DEFAULT_SITE_ID,
        asset_id=asset_val if asset_val is not None else DEFAULT_ASSET_ID,
        sensor_values=signal_values,
        customer_id=customer_id,
    )
    return frame


def normalize_external_batch_payload(
    payload: Mapping[str, Any],
    *,
    customer_id: str | None = None,
) -> tuple[list[dict[str, Any]], MappingReview]:
    if not isinstance(payload, Mapping):
        raise ValueError("unsupported_payload_shape: batch payload must be an object")

    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = payload.get("records")
    if raw_items is None:
        raw_items = payload.get("payloads")
    if raw_items is None and isinstance(payload.get("stream"), Mapping):
        raw_items = payload.get("stream", {}).get("records")
    if raw_items is None:
        raise ValueError("unsupported_payload_shape: expected items/records/payloads or stream.records")
    if not isinstance(raw_items, list):
        raise ValueError("unsupported_payload_shape: batch records must be a list")

    mapping = payload.get("mapping") if isinstance(payload.get("mapping"), Mapping) else {}
    review = MappingReview(
        inferred_mapping={"mapping": dict(mapping)},
        warnings=[],
        requires_confirmation=False,
    )
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        try:
            normalized.append(
                normalize_external_payload(
                    item,
                    customer_id=customer_id,
                    mapping=mapping,
                    require_confirmed_mapping=False,
                )
            )
        except ValueError as exc:
            raise ValueError(f"batch_item_{idx}: {exc}") from exc
    return normalized, review
