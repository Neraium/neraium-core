from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, MutableMapping


@dataclass(frozen=True)
class IngestRouterDependencies:
    service_instance: Any
    require_api_key: Callable[..., None]
    resolve_customer_id: Callable[[str | None], str]
    resolve_run_id_with_default: Callable[..., str]
    actionable_validation_detail: Callable[[str], str]
    normalize_external_payload: Callable[..., dict[str, Any]]
    normalize_external_batch_payload: Callable[..., tuple[list[dict[str, Any]], list[Any]]]
    normalize_canonical_records_payload: Callable[..., list[dict[str, Any]]]
    process_alerts_after_ingest: Callable[..., list[dict[str, Any]]]
    results_envelope: Callable[[list[dict[str, Any]], dict[str, Any] | None], dict[str, Any]]
    parse_csv_rows: Callable[..., tuple[list[str], list[dict[str, Any]], list[Any]]]
    infer_csv_mapping_stage: Callable[..., Any]
    validate_csv_mapping_stage: Callable[..., list[Any]]
    issue_to_dict: Callable[[Any], dict[str, Any]]
    request_body_limit: int
    normalize_content_length: Callable[[Any], int | None]
    stream_upload_to_tempfile: Callable[..., Any]
    update_ingest_job: Callable[..., None]
    public_ingest_job: Callable[[dict[str, Any]], dict[str, Any]]
    persist_operational_state: Callable[..., None]
    start_ingest_job_worker: Callable[..., None]
    ingest_jobs: MutableMapping[str, dict[str, Any]]
    ingest_jobs_lock: Lock
    utc_now_iso: Callable[[], str]


@dataclass(frozen=True)
class DemoRouterDependencies:
    service_instance: Any
    require_api_key: Callable[..., None]
    resolve_customer_id: Callable[[str | None], str]
    resolve_run_id_with_default: Callable[..., str]
    start_demo_seed_job: Callable[..., None]
    public_demo_job: Callable[[dict[str, Any]], dict[str, Any]]
    demo_jobs: MutableMapping[str, dict[str, Any]]
    demo_jobs_lock: Lock
    load_greenhouse_demo_subset: Callable[[int], list[dict[str, Any]]]
    log_structured: Callable[..., None]
    summarize_exception_for_logs: Callable[[Exception], str]
    utc_now_iso: Callable[[], str]


@dataclass(frozen=True)
class IntegrationsRouterDependencies:
    app: Any
    require_api_key: Callable[..., None]
    resolve_customer_id: Callable[[str | None], str]
    resolve_run_id_with_default: Callable[..., str]
    service_instance: Any
    pull_manager: Any
    log_structured: Callable[..., None]


@dataclass(frozen=True)
class OnboardingRouterDependencies:
    resolve_customer_id: Callable[[str | None], str]
    service_instance: Any
    normalize_external_payload: Callable[..., dict[str, Any]]
    is_api_key_valid: Callable[[str | None, str | None], bool]
    configured_api_key: str | None
    ensure_default_run: Callable[..., dict[str, Any]]
    persist_operational_state: Callable[..., None]
    store_instance: Any | None
    persisted_state_enabled: bool
