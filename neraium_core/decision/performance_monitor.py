"""Performance monitoring for the decision engine.

Tracks latency, cache hit rates, and memory usage for optimization validation.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class PhaseMetrics:
    """Metrics for a single computation phase."""
    phase_name: str
    call_count: int = 0
    total_time: float = 0.0
    cache_hits: int = 0
    avg_time: float = 0.0
    hit_rate: float = 0.0

    def update(self, elapsed: float, was_cached: bool = False) -> None:
        """Update metrics with a new call."""
        self.call_count += 1
        self.total_time += elapsed
        if was_cached:
            self.cache_hits += 1
        self.avg_time = self.total_time / self.call_count if self.call_count > 0 else 0.0
        self.hit_rate = self.cache_hits / self.call_count if self.call_count > 0 else 0.0


class PerformanceMonitor:
    """Tracks decision engine performance metrics.

    Usage:
        monitor = PerformanceMonitor()
        with monitor.phase("severity_classification"):
            # compute severity
    """

    def __init__(self):
        self._phases: dict[str, PhaseMetrics] = defaultdict(
            lambda: PhaseMetrics(phase_name="")
        )
        self._frame_times: list[float] = []
        self._frame_start: Optional[float] = None

    def start_frame(self) -> None:
        """Mark start of frame processing."""
        self._frame_start = time.perf_counter()

    def end_frame(self) -> float:
        """Mark end of frame, return elapsed time."""
        if self._frame_start is None:
            return 0.0
        elapsed = time.perf_counter() - self._frame_start
        self._frame_times.append(elapsed)
        return elapsed

    class _PhaseContext:
        """Context manager for timing a phase."""

        def __init__(self, monitor: PerformanceMonitor, phase_name: str, was_cached: bool = False):
            self.monitor = monitor
            self.phase_name = phase_name
            self.was_cached = was_cached
            self.start_time = 0.0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.perf_counter() - self.start_time
            metrics = self.monitor._phases[self.phase_name]
            metrics.phase_name = self.phase_name
            metrics.update(elapsed, self.was_cached)

    def phase(self, name: str, was_cached: bool = False) -> _PhaseContext:
        """Context manager to time a phase."""
        return self._PhaseContext(self, name, was_cached)

    def get_phase_stats(self, phase_name: Optional[str] = None) -> dict[str, Any]:
        """Get statistics for a phase or all phases."""
        if phase_name:
            if phase_name not in self._phases:
                return {}
            m = self._phases[phase_name]
            return {
                "calls": m.call_count,
                "total_ms": m.total_time * 1000,
                "avg_ms": m.avg_time * 1000,
                "cache_hits": m.cache_hits,
                "hit_rate": m.hit_rate,
            }

        return {
            name: {
                "calls": m.call_count,
                "total_ms": m.total_time * 1000,
                "avg_ms": m.avg_time * 1000,
                "cache_hits": m.cache_hits,
                "hit_rate": m.hit_rate,
            }
            for name, m in self._phases.items()
        }

    def get_frame_stats(self) -> dict[str, float]:
        """Get frame processing time statistics."""
        if not self._frame_times:
            return {}
        times = self._frame_times
        return {
            "frames_processed": len(times),
            "total_time_ms": sum(times) * 1000,
            "avg_time_ms": (sum(times) / len(times)) * 1000 if times else 0.0,
            "min_time_ms": min(times) * 1000 if times else 0.0,
            "max_time_ms": max(times) * 1000 if times else 0.0,
            "p95_time_ms": sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) > 1 else 0.0,
        }

    def report(self) -> str:
        """Generate a human-readable performance report."""
        frame_stats = self.get_frame_stats()
        phase_stats = self.get_phase_stats()

        lines = ["=== Frame Processing ==="]
        lines.append(f"Frames: {frame_stats.get('frames_processed', 0)}")
        lines.append(f"Avg frame time: {frame_stats.get('avg_time_ms', 0):.2f}ms")
        lines.append(f"P95 latency: {frame_stats.get('p95_time_ms', 0):.2f}ms")
        lines.append("")

        lines.append("=== Phase Breakdown ===")
        for phase_name in sorted(phase_stats.keys()):
            stats = phase_stats[phase_name]
            lines.append(
                f"{phase_name}: avg {stats['avg_ms']:.2f}ms, "
                f"hit_rate {stats['hit_rate']:.1%} "
                f"({stats['cache_hits']}/{stats['calls']} calls)"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all metrics."""
        self._phases.clear()
        self._frame_times.clear()
