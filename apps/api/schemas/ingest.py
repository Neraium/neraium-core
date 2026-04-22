from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from neraium_core.ingestion_normalization import normalize_external_batch_payload, normalize_external_payload


class IngestRequest(BaseModel):
    model_config = {"extra": "allow"}

    customer_id: str | None = None
    timestamp: str | None = None
    site_id: str | None = None
    asset_id: str | None = None
    sensor_values: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = normalize_external_payload(data, customer_id=data.get("customer_id"))
        return {
            "customer_id": normalized.get("customer_id"),
            "timestamp": normalized.get("timestamp"),
            "site_id": normalized.get("site_id"),
            "asset_id": normalized.get("asset_id"),
            "sensor_values": normalized.get("sensor_values", {}),
        }


class IngestFrameRequest(BaseModel):
    """Production API payload for a single telemetry frame."""

    timestamp: str
    site_id: str
    asset_id: str
    sensor_values: dict[str, Any] = Field(default_factory=dict)
    customer_id: str | None = None


CMAPSS_REPLAY_DEFAULT_MAX_FRAMES = 240


class DemoSeedRequest(BaseModel):
    run_id: str | None = None
    customer_id: str | None = None
    profile: Literal["sample", "stable", "watch", "critical"] = "sample"
    minutes: int = Field(default=120, ge=10, le=240)
    site_id: str = "demo-site"
    asset_id: str = "demo-asset"


class DemoCmapssStartRequest(BaseModel):
    customer_id: str | None = None
    max_frames: int = Field(default=CMAPSS_REPLAY_DEFAULT_MAX_FRAMES, ge=30, le=500)


class BatchIngestRequest(BaseModel):
    items: list[IngestRequest]

    @model_validator(mode="before")
    @classmethod
    def _accept_records_alias(cls, data: Any) -> Any:
        """Accept legacy/front-end payloads that send `records` instead of `items`."""
        if isinstance(data, dict):
            if "items" in data:
                return data
            if any(k in data for k in ("records", "payloads", "stream")):
                normalized, _ = normalize_external_batch_payload(
                    data,
                    customer_id=data.get("customer_id"),
                )
                remapped = dict(data)
                remapped["items"] = normalized
                return remapped
        return data


class JsonIngestRequest(BaseModel):
    model_config = {"extra": "allow"}

    customer_id: str | None = None
    mapping: dict[str, Any] | None = None
    items: list[dict[str, Any]] | None = None
    records: list[dict[str, Any]] | None = None
    payloads: list[dict[str, Any]] | None = None


class CanonicalIngestRequest(BaseModel):
    customer_id: str | None = None
    records: list[dict[str, Any]] | None = None
    items: list[dict[str, Any]] | None = None


class CsvColumnMappingPayload(BaseModel):
    """Semantic roles: which CSV columns map to time, entity, optional site, and numeric sensors."""

    timestamp: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    site_id: str | None = None
    sensor_columns: list[str] = Field(min_length=1)


class CsvIngestRequest(BaseModel):
    customer_id: str | None = None
    csv_text: str
    column_mapping: CsvColumnMappingPayload | None = None


class CsvPreviewRequest(BaseModel):
    csv_sample: str = Field(..., max_length=524_288)


class CsvPreviewResponse(BaseModel):
    headers: list[str]
    suggested_mapping: dict[str, Any] | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issue_details: list[dict[str, Any]] = Field(default_factory=list)
    warning_details: list[dict[str, Any]] = Field(default_factory=list)
    requires_confirmation: bool = False
    preview_state: str = "preview_ready"


class IngestJobEnvelope(BaseModel):
    job_id: str
    status: str
    run_id: str | None = None
    customer_id: str
    filename: str
    created_at: str
    updated_at: str
    rows_processed: int = 0
    rows_succeeded: int = 0
    rows_failed: int = 0
    partial_success: bool = False
    upload_bytes_received: int = 0
    upload_bytes_total: int | None = None
    error_samples: list[dict[str, Any]] = Field(default_factory=list)
    message: str | None = None
    latest_result: dict[str, Any] | None = None
    lifecycle_phase: str | None = None
    ui_state: str | None = None
    terminal_state: str | None = None
    failure_category: str | None = None
