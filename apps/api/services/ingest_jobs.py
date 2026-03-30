from __future__ import annotations

from typing import Any


class IngestJobState:
    """Shared ingest job state container for upload + background processing."""

    def __init__(self, ingest_jobs: dict[str, Any], ingest_jobs_lock: Any) -> None:
        self.ingest_jobs = ingest_jobs
        self.ingest_jobs_lock = ingest_jobs_lock

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.ingest_jobs_lock:
            job = self.ingest_jobs.get(job_id)
            return dict(job) if isinstance(job, dict) else None

    def set(self, job_id: str, payload: dict[str, Any]) -> None:
        with self.ingest_jobs_lock:
            self.ingest_jobs[job_id] = dict(payload)

    def update(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.ingest_jobs_lock:
            job = self.ingest_jobs.get(job_id)
            if job is None:
                return None
            job.update(fields)
            return dict(job)
