from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeStateStore:
    """In-memory operational state containers and their synchronization primitives."""

    ingest_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    demo_jobs: dict[str, dict[str, Any]] = field(default_factory=dict)
    pull_integrations: dict[str, dict[str, Any]] = field(default_factory=dict)
    alerts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    cmapss_fd004_cache: dict[int, list[dict[str, Any]]] = field(default_factory=dict)

    ingest_jobs_lock: threading.Lock = field(default_factory=threading.Lock)
    demo_jobs_lock: threading.Lock = field(default_factory=threading.Lock)
    pull_integrations_lock: threading.Lock = field(default_factory=threading.Lock)
    alerts_lock: threading.Lock = field(default_factory=threading.Lock)
    cmapss_fd004_cache_lock: threading.Lock = field(default_factory=threading.Lock)
