from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PullIntegrationStartRequest(BaseModel):
    endpoint_url: str | None = Field(default=None, min_length=1, max_length=2000)
    polling_interval_seconds: float | None = Field(default=None, ge=0.2, le=3600.0)
    auth_type: Literal["none", "basic", "bearer"] | None = None
    username: str | None = None
    password: str | None = None
    token: str | None = None
    run_id: str | None = None
    retry_max_attempts: int | None = Field(default=None, ge=1, le=10)
    retry_backoff_seconds: float | None = Field(default=None, ge=0.05, le=60.0)
    request_timeout_seconds: float | None = Field(default=None, ge=1.0, le=120.0)


class PullIntegrationStatusEnvelope(BaseModel):
    customer_id: str
    endpoint_url: str | None = None
    run_id: str | None = None
    auth_type: str = "none"
    running: bool
    status: str
    polling_interval_seconds: float | None = None
    retry_max_attempts: int | None = None
    retry_backoff_seconds: float | None = None
    request_timeout_seconds: float | None = None
    started_at: str | None = None
    updated_at: str | None = None
    last_poll_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    last_http_status: int | None = None
    total_polls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    total_ingested: int = 0
    message: str | None = None
