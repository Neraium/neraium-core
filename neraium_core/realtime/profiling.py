"""Stage timing for incremental pipeline introspection."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def profiling_enabled() -> bool:
    import os

    v = os.environ.get("NERAIUM_PROFILE_INCREMENTAL", "")
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class StageTimer:
    """Accumulates per-stage wall times (seconds)."""

    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    last: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()
        self.last.clear()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            self.last[name] = dt

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in self.totals:
            n = max(1, self.counts.get(k, 1))
            out[k] = {
                "total_s": float(self.totals[k]),
                "count": int(self.counts.get(k, 0)),
                "mean_ms": float(1000.0 * self.totals[k] / n),
                "last_ms": float(1000.0 * self.last.get(k, 0.0)),
            }
        worst = None
        worst_ms = -1.0
        for k, v in out.items():
            m = float(v["mean_ms"])
            if m > worst_ms:
                worst_ms = m
                worst = k
        return {"stages": out, "worst_stage": worst, "worst_mean_ms": worst_ms}


def run_timed(name: str, timer: StageTimer, fn: Callable[[], Any]) -> Any:
    with timer.stage(name):
        return fn()
