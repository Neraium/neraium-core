from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


_KEY_RE = re.compile(r"[^a-z0-9_]+")


class IntegrationConfigError(ValueError):
    """Raised when integration config is invalid."""


class IntegrationMappingError(ValueError):
    """Raised when payload mapping cannot produce ingestable rows."""


def canonical_key(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = _KEY_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(override, Mapping):
        return dict(base)
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _env_defaults() -> dict[str, Any]:
    auth_type = canonical_key(
        os.getenv(
            "NERAIUM_DEFAULT_CUSTOMER_API_AUTH_TYPE",
            os.getenv("NERAIUM_CUSTOMER_API_AUTH_TYPE", "none"),
        )
    ) or "none"
    if auth_type not in {"none", "basic", "bearer"}:
        auth_type = "none"
    return {
        "mode": canonical_key(os.getenv("NERAIUM_CUSTOMER_API_MODE", "pull")) or "pull",
        "endpoint_url": os.getenv(
            "NERAIUM_DEFAULT_CUSTOMER_API_BASE_URL",
            os.getenv("NERAIUM_CUSTOMER_API_BASE_URL"),
        ),
        "auth_type": auth_type,
        "username": os.getenv(
            "NERAIUM_DEFAULT_CUSTOMER_API_USERNAME",
            os.getenv("NERAIUM_CUSTOMER_API_USERNAME"),
        ),
        "password": os.getenv(
            "NERAIUM_DEFAULT_CUSTOMER_API_PASSWORD",
            os.getenv("NERAIUM_CUSTOMER_API_PASSWORD"),
        ),
        "token": os.getenv(
            "NERAIUM_DEFAULT_CUSTOMER_API_TOKEN",
            os.getenv("NERAIUM_CUSTOMER_API_TOKEN"),
        ),
        "polling_interval_seconds": max(
            0.2,
            _as_float(
                os.getenv(
                    "NERAIUM_PULL_POLLING_INTERVAL_SECONDS",
                    os.getenv("NERAIUM_CUSTOMER_API_POLL_INTERVAL_SECONDS"),
                ),
                30.0,
            ),
        ),
        "retry_max_attempts": max(
            1,
            _as_int(
                os.getenv(
                    "NERAIUM_PULL_RETRY_MAX_ATTEMPTS",
                    os.getenv("NERAIUM_CUSTOMER_API_RETRY_MAX_ATTEMPTS"),
                ),
                3,
            ),
        ),
        "retry_backoff_seconds": max(
            0.05,
            _as_float(
                os.getenv(
                    "NERAIUM_PULL_RETRY_BACKOFF_SECONDS",
                    os.getenv("NERAIUM_CUSTOMER_API_RETRY_BACKOFF_SECONDS"),
                ),
                1.0,
            ),
        ),
        "request_timeout_seconds": max(
            1.0,
            _as_float(
                os.getenv(
                    "NERAIUM_PULL_REQUEST_TIMEOUT_SECONDS",
                    os.getenv("NERAIUM_CUSTOMER_API_TIMEOUT_SECONDS"),
                ),
                10.0,
            ),
        ),
        "mapping": {
            "field_aliases": {},
            "sensor_aliases": {},
            "defaults": {},
            "ignore_fields": [],
            "payload_path": "",
            "items_path": "",
        },
        "headers": {},
    }


def default_integration_config() -> dict[str, Any]:
    return {"default": {}, "customers": {}}


def load_integration_config(path: str | None) -> dict[str, Any]:
    if not path:
        return default_integration_config()
    p = Path(path)
    if not p.exists():
        return default_integration_config()
    try:
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise IntegrationConfigError(f"Invalid JSON in integration config: {p}") from exc
    except OSError as exc:
        raise IntegrationConfigError(f"Unable to read integration config: {p}") from exc
    if not isinstance(data, dict):
        raise IntegrationConfigError("Integration config must be a JSON object.")
    out = default_integration_config()
    out.update(data)
    if not isinstance(out.get("default"), dict):
        raise IntegrationConfigError("Integration config 'default' must be an object.")
    if not isinstance(out.get("customers"), dict):
        raise IntegrationConfigError("Integration config 'customers' must be an object.")
    return out


def resolve_customer_integration(
    *,
    customer_id: str,
    config_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved_customer = str(customer_id or "").strip() or "default-customer"
    env_cfg = _env_defaults()
    doc = config_doc or default_integration_config()
    default_cfg = doc.get("default") if isinstance(doc, dict) else {}
    customers_cfg = doc.get("customers") if isinstance(doc, dict) else {}
    customer_cfg = customers_cfg.get(resolved_customer) if isinstance(customers_cfg, dict) else {}
    merged = _deep_merge(env_cfg, default_cfg if isinstance(default_cfg, dict) else {})
    merged = _deep_merge(merged, customer_cfg if isinstance(customer_cfg, dict) else {})
    merged["customer_id"] = resolved_customer
    return merged


def _alias_map(mapping: Mapping[str, Any] | None) -> dict[str, list[str]]:
    defaults: dict[str, list[str]] = {
        "timestamp": ["timestamp", "time", "event_time", "event_timestamp", "ts", "datetime"],
        "site_id": ["site_id", "site", "siteid", "location", "plant", "facility"],
        "asset_id": ["asset_id", "asset", "assetid", "device_id", "machine_id", "unit"],
        "sensor_values": ["sensor_values", "sensors", "signals", "metrics", "values", "telemetry"],
    }
    if not isinstance(mapping, Mapping):
        return defaults
    aliases = mapping.get("field_aliases")
    if not isinstance(aliases, Mapping):
        return defaults
    out = {k: list(v) for k, v in defaults.items()}
    for key, raw in aliases.items():
        canon = canonical_key(key)
        if canon not in out:
            continue
        vals: list[str] = []
        if isinstance(raw, list):
            vals = [canonical_key(x) for x in raw if canonical_key(x)]
        elif isinstance(raw, str):
            c = canonical_key(raw)
            vals = [c] if c else []
        if vals:
            out[canon] = list(dict.fromkeys(vals + out[canon]))
    return out


def _extract_path(payload: Any, path: str) -> Any:
    text = str(path or "").strip()
    if not text:
        return payload
    current = payload
    for part in [p for p in text.split(".") if p]:
        if isinstance(current, Mapping):
            if part not in current:
                raise IntegrationMappingError(f"Path '{text}' not found in payload.")
            current = current[part]
            continue
        if isinstance(current, list):
            try:
                idx = int(part)
            except ValueError as exc:
                raise IntegrationMappingError(f"Path '{text}' expects list index at '{part}'.") from exc
            if idx < 0 or idx >= len(current):
                raise IntegrationMappingError(f"Path '{text}' index out of range: {idx}.")
            current = current[idx]
            continue
        raise IntegrationMappingError(f"Path '{text}' is invalid at '{part}'.")
    return current


def normalize_external_payload(
    payload: Any,
    *,
    customer_id: str,
    mapping: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]]
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        if not all(isinstance(item, dict) for item in payload["items"]):
            raise IntegrationMappingError("Payload items[] must contain JSON objects.")
        rows = [dict(item) for item in payload["items"]]
    elif isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise IntegrationMappingError("Payload list must contain JSON objects.")
        rows = [dict(item) for item in payload]
    elif isinstance(payload, dict):
        rows = [dict(payload)]
    else:
        raise IntegrationMappingError("Payload must be an object, list, or {items:[...]} wrapper.")

    if not rows:
        return []

    mapping_dict = dict(mapping or {})
    alias_map = _alias_map(mapping_dict)
    defaults = mapping_dict.get("defaults") if isinstance(mapping_dict.get("defaults"), Mapping) else {}
    sensor_aliases_raw = (
        mapping_dict.get("sensor_aliases")
        if isinstance(mapping_dict.get("sensor_aliases"), Mapping)
        else {}
    )
    sensor_aliases = {canonical_key(k): canonical_key(v) for k, v in sensor_aliases_raw.items()}
    ignore_fields = mapping_dict.get("ignore_fields") if isinstance(mapping_dict.get("ignore_fields"), list) else []
    ignored = {canonical_key(v) for v in ignore_fields}

    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        canon_to_value: dict[str, Any] = {}
        for k, v in row.items():
            ck = canonical_key(k)
            if ck:
                canon_to_value.setdefault(ck, v)

        def _pick(field_name: str) -> Any:
            for candidate in alias_map[field_name]:
                if candidate in canon_to_value:
                    return canon_to_value[candidate]
            return None

        timestamp = _pick("timestamp")
        site_id = _pick("site_id")
        asset_id = _pick("asset_id")
        sensor_values_raw = _pick("sensor_values")
        if site_id in {None, ""} and defaults:
            site_id = defaults.get("site_id")
        if asset_id in {None, ""} and defaults:
            asset_id = defaults.get("asset_id")

        missing = [
            name
            for name, value in [("timestamp", timestamp), ("site_id", site_id), ("asset_id", asset_id)]
            if value in {None, ""}
        ]
        if missing:
            raise IntegrationMappingError(
                f"Row {row_index}: missing required fields after mapping: {', '.join(missing)}."
            )

        sensor_values: dict[str, Any] = {}
        if isinstance(sensor_values_raw, Mapping):
            for key, value in sensor_values_raw.items():
                ckey = canonical_key(key)
                if not ckey:
                    continue
                mapped = sensor_aliases.get(ckey, ckey)
                if mapped:
                    sensor_values[mapped] = value
        elif sensor_values_raw not in {None, ""}:
            raise IntegrationMappingError(
                f"Row {row_index}: mapped sensor_values field must be an object."
            )
        else:
            reserved: set[str] = set()
            for names in alias_map.values():
                reserved.update(names)
            reserved.update({"customer_id", "run_id"})
            reserved.update(ignored)
            for ck, value in canon_to_value.items():
                if ck in reserved:
                    continue
                mapped = sensor_aliases.get(ck, ck)
                if mapped:
                    sensor_values[mapped] = value

        if not sensor_values:
            raise IntegrationMappingError(f"Row {row_index}: no sensor fields found after mapping.")

        out.append(
            {
                "customer_id": str(customer_id or "").strip() or "default-customer",
                "timestamp": timestamp,
                "site_id": site_id,
                "asset_id": asset_id,
                "sensor_values": sensor_values,
            }
        )

    return out


def apply_integration_mapping(
    payload: Any,
    *,
    customer_id: str,
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    resolved = resolve_customer_integration(customer_id=customer_id, config_doc=config)
    mapping_cfg = dict(resolved.get("mapping") or {})

    # Backward/lenient config support: allow mapping keys at customer root.
    for key in ("field_aliases", "sensor_aliases", "defaults", "ignore_fields", "payload_path", "items_path"):
        if key in resolved and key not in mapping_cfg:
            mapping_cfg[key] = resolved.get(key)

    source = payload
    payload_path = str(mapping_cfg.get("payload_path") or "").strip()
    items_path = str(mapping_cfg.get("items_path") or "").strip()
    if payload_path:
        source = _extract_path(source, payload_path)
    if items_path:
        items = _extract_path(source, items_path)
        source = {"items": items}

    return normalize_external_payload(source, customer_id=resolved["customer_id"], mapping=mapping_cfg)
